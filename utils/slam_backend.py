import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.slam_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from gaussian_splatting.utils.graphics_utils import rotation2euler, getWorld2View2, saveDict2ckpt
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.slam_utils import compute_point_ratio
import faiss


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = GaussianModel(0)

        # add submap
        self.sub_gaussians = GaussianModel(0)
        self.new_submap_frame_ids = []
        self.submap_rot_thre = 100
        self.submap_trans_thre = 3
        self.submap_id = 0
        self.iteration_count_sub = 0
        self.global_keyframes = []
        self.sub_keyframes = []
        self.has_global = False
        self.iter_per_kf = 10
        self.submap_size_threshold = 100

        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]

    def add_next_kf_submap(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.sub_keyframes.append(frame_idx)
        self.sub_gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def add_next_kf_global(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.global_keyframes.append(frame_idx)
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.keyframe_optimizers = None

        self.iteration_count_sub = 0
        self.new_submap_frame_ids = []
        self.global_keyframes = []
        self.has_global = False

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

        # remove current submap
        self.sub_gaussians.prune_points(self.sub_gaussians.unique_kfIDs >= 0)

        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_submap(self, cur_frame_idx, viewpoint, iters=None):
        self.new_submap_frame_ids.append(cur_frame_idx)
        if iters is None:
            iters = self.init_itr_num
        for mapping_iteration in range(iters):
            self.iteration_count_sub += 1
            render_pkg = render(viewpoint, self.sub_gaussians, self.pipeline_params, self.background)
            
            # get the render results
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            depth = render_pkg["depth"]
            opacity = render_pkg["opacity"]
            n_touched = render_pkg["n_touched"]

            # get the loss for the mapping
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.sub_gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.sub_gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.sub_gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.sub_gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count_sub == self.init_gaussian_reset or (
                    self.iteration_count_sub == self.opt_params.densify_from_iter
                ):
                    self.sub_gaussians.reset_opacity()

                self.sub_gaussians.optimizer.step()
                self.sub_gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized submap")
        return render_pkg

    def exceeds_motion_thresholds(self, cur_frame_idx, last_frame_idx, trans_thre=1, rot_thre=45):
        last_camera = self.viewpoints[last_frame_idx]
        cur_camera = self.viewpoints[cur_frame_idx]

        cur_CW = getWorld2View2(cur_camera.R, cur_camera.T) # (4, 4)
        last_CW = getWorld2View2(last_camera.R, last_camera.T) # (4, 4)
        last_WC = torch.linalg.inv(last_CW)
        
        delta_pose = torch.matmul(cur_CW.float(), last_WC.float())
        trans_diff = torch.norm(delta_pose[:3, 3]) # distance between two frames' translation
        rot_diff_deg = torch.abs(rotation2euler(delta_pose[:3, :3])) # shape: (3,)
        
        exceeds_thresholds = (trans_diff > trans_thre)
        return exceeds_thresholds.item()


    def should_start_new_submap(self, cur_frame_idx):
        if self.exceeds_motion_thresholds(cur_frame_idx, self.new_submap_frame_ids[-1], self.submap_trans_thre, self.submap_rot_thre):
            return True
        elif self.has_global and self.check_loss(cur_frame_idx) and abs(self.sub_keyframes.index(cur_frame_idx) - self.sub_keyframes.index(
            self.new_submap_frame_ids[-1])) > self.submap_size_threshold * 0.5:
            return True
        elif abs(self.sub_keyframes.index(cur_frame_idx) - self.sub_keyframes.index(
            self.new_submap_frame_ids[-1])) > self.submap_size_threshold:
            return True
        return False

    def check_loss(self, cur_frame_idx):
        with torch.no_grad():
            # get the loss for submap
            render_pkg = render(
                self.viewpoints[cur_frame_idx], self.sub_gaussians, self.pipeline_params, self.background
            )
            image = render_pkg["render"]
            depth = render_pkg["depth"]
            opacity = render_pkg["opacity"]
            submap_loss_mapping = get_loss_mapping(
                self.config, image, depth, self.viewpoints[cur_frame_idx], opacity
            )
            
            # get the loss for global map
            render_pkg = render(
                self.viewpoints[cur_frame_idx], self.gaussians, self.pipeline_params, self.background
            )
            image = render_pkg["render"]
            depth = render_pkg["depth"]
            opacity = render_pkg["opacity"]
            loss_mapping = get_loss_mapping(
                self.config, image, depth, self.viewpoints[cur_frame_idx], opacity
            )
            
            check = (loss_mapping < submap_loss_mapping) and (loss_mapping < 0.005)
            return check
        
    
    def start_new_submap(self, cur_frame_idx, viewpoint):
        # save current submap
        output_path = self.config["Results"]["save_dir"] + "/submaps"
        gaussian_params = self.sub_gaussians.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(self.sub_keyframes)
        }
        mkdir_p(output_path)
        saveDict2ckpt(submap_ckpt, f"sub_{submap_ckpt_name}.ckpt", directory=output_path)

        # merge submaps into the global map
        self.merge_submaps(cur_frame_idx)
        
        self.submap_id += 1
        self.iteration_count_sub = 0
        self.sub_keyframes = [cur_frame_idx]
        self.current_window = [cur_frame_idx]

        # create neighbor global frames
        neighbor_frame = self.get_neighbor_frame(cur_frame_idx)
        self.map(neighbor_frame, self.gaussians, iters=100, type="global", isBA=False, useSSIM=False)

        # create new submap
        # self.sub_gaussians = GaussianModel(0, config=self.config)
        # self.sub_gaussians.init_lr(6.0)
        # self.sub_gaussians.training_setup(self.opt_params)
        # prune points that are not in neighbor frames
        self.sub_gaussians.prune_points(self.sub_gaussians.unique_kfIDs != cur_frame_idx)

        # add the gaussians that are visible in the current frame in global map to the submap
        for idx in neighbor_frame:
            self.camera_gaussians_to_submap(idx)
        # sort neighbor frames by index and get the latest 10 frames
        self.current_window = sorted(neighbor_frame, reverse=True)
        self.initialize_submap(cur_frame_idx, viewpoint, iters=10)
        
    def get_neighbor_frame(self, cur_frame_idx):
        with torch.no_grad():
            frame = []
            for idx in self.global_keyframes:
                if not self.exceeds_motion_thresholds(cur_frame_idx, idx,trans_thre=0.6):
                    frame.append(idx)
        return frame
    
    def merge_submaps(self, current_frame_idx=None):
        for frame_idx in self.sub_keyframes:
            if current_frame_idx is not None and frame_idx == current_frame_idx:
                continue
            self.global_keyframes.append(frame_idx)
        # merge submaps into the global map
        self.gaussians.densification_postfix(
            self.sub_gaussians._xyz,
            self.sub_gaussians._features_dc,
            self.sub_gaussians._features_rest,
            self.sub_gaussians._opacity,
            self.sub_gaussians._scaling,
            self.sub_gaussians._rotation,
            new_kf_ids=self.sub_gaussians.unique_kfIDs,
            new_n_obs=self.sub_gaussians.n_obs,
        )

        self.has_global = True
        
    def merge_submaps_with_neighbor(self, radius: float = 0.0001, device: str = "cuda"):
        Log("Merging submaps with neighbor")
        pts_index = faiss.IndexFlatL2(3)
        if device == "cuda":
            d = 3  # Dimension of the points
            nlist = 500  # Number of clusters
            # Setting up GPU resources and the GPU index
            res = faiss.StandardGpuResources()  # Allocating GPU resources
            # Create the index directly on GPU
            pts_index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
            pts_index.nprobe = 5

            # add all submaps to the global map
        self.gaussians.densification_postfix(
            self.sub_gaussians._xyz,
            self.sub_gaussians._features_dc,
            self.sub_gaussians._features_rest,
            self.sub_gaussians._opacity,
            self.sub_gaussians._scaling,
            self.sub_gaussians._rotation,
            new_kf_ids=self.sub_gaussians.unique_kfIDs,
            new_n_obs=self.sub_gaussians.n_obs,
        )

        # clean global map by removing points that have neighbors within the radius
        # Prepare the points for FAISS processing
        current_pts = self.gaussians._xyz.to(device).float().detach()
        pts_index.train(current_pts.cpu().numpy())

        # speed up the search by splitting the point cloud because the memory limitation of the GPU is 65535
        split_pos = torch.split(current_pts, 65535, dim=0)
        distances_list = []
        for split_p in split_pos:
            distance, id = pts_index.search(split_p.detach().cpu().numpy(), 8)  # search for 8 nearest neighbors
            distances_list.append(torch.from_numpy(distance))

        distances = torch.cat(distances_list, dim=0)

        neighbor_num = (distances < radius).sum(axis=1).int()
        ids_to_include = torch.where(neighbor_num == 0)[0]  # points that have no neighbors within the radius, [0] mean the row index

        # delete gaussian from map
        invalid_mask = torch.ones(current_pts.shape[0], dtype=torch.bool)  # shape: (n_points,)
        invalid_mask[ids_to_include] = False  # set the valid points to False, so that all other points will be removed
        invalid_mask = invalid_mask.squeeze()  # remove the dimension of 1
        self.gaussians.prune_points(invalid_mask)  # remove invalid points from the global map
        
        self.has_global = True
    
    def camera_gaussians_to_submap(self, kf_idx):
        ids_mask = self.gaussians.unique_kfIDs == kf_idx
        self.sub_gaussians.densification_postfix(
            self.gaussians._xyz[ids_mask],
            self.gaussians._features_dc[ids_mask],
            self.gaussians._features_rest[ids_mask],
            self.gaussians._opacity[ids_mask],
            self.gaussians._scaling[ids_mask],
            self.gaussians._rotation[ids_mask],
            new_kf_ids=self.gaussians.unique_kfIDs[ids_mask],
            new_n_obs=self.gaussians.n_obs[ids_mask],
        )
        
    def get_sub_window(self, current_window : list):
        current_sub_window = []
        for f in range(len(current_window)):
            if current_window[f] >= self.new_submap_frame_ids[-1]:
                current_sub_window.append(current_window[f])
        return current_sub_window
    
    def create_keyframe_optimizers(self, current_window):
        opt_params = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        for cam_idx in range(len(current_window)):
            if current_window[cam_idx] == 0:  # skip the first keyframe (it is the reference frame)
                continue
            if self.has_global and cam_idx == 0:
                continue
            viewpoint = self.viewpoints[current_window[cam_idx]]
            if cam_idx < frames_to_optimize:
                opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                              * 0.5,
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": self.config["Training"]["lr"][
                                  "cam_trans_delta"
                              ]
                              * 0.5,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(viewpoint.uid),
                }
            )
        return opt_params

    def map(self, current_window, gaussians: GaussianModel, noSplit=False, iters=1, isBA= False, type="global", useSSIM=True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        for _ in range(iters):
            if type == "global":
                self.iteration_count += 1
                iter_count = self.iteration_count
            else:
                self.iteration_count_sub += 1
                iter_count = self.iteration_count_sub
                self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, gaussians, self.pipeline_params, self.background
                )
                
                # get the render results
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                depth = render_pkg["depth"]
                opacity = render_pkg["opacity"]
                n_touched = render_pkg["n_touched"]

                # get the loss for the mapping
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, gaussians, self.pipeline_params, self.background
                )
                
                # get the render results
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                depth = render_pkg["depth"]
                opacity = render_pkg["opacity"]
                n_touched = render_pkg["n_touched"]
                
                # get the loss for the mapping
                if useSSIM:
                    loss = get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity
                    )
                    gt_image = viewpoint.original_image.cuda()
                    loss_mapping += (1.0 - self.opt_params.lambda_dssim) * (loss) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                else:
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity
                    )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                if type == "submap":
                    self.occ_aware_visibility = occ_aware_visibility

                if noSplit:
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    iter_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (iter_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log(f"Resetting the opacity of non-visible Gaussians in {type}")
                    gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iter_count)
                
                if isBA:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    # Pose update
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self, iters=26000):
        for iteration in range(1, iters + 1):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone())) # tuple of keyframe index, rotation, translation
        if tag is None:
            tag = "sync_backend"
        if tag is "color_refinement":
            msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes, self.current_window]
        elif tag is "end":
            msg = [tag, clone_obj(self.gaussians)]
        else:
            msg = [tag, clone_obj(self.sub_gaussians), self.occ_aware_visibility, keyframes, self.current_window]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window, self.sub_gaussians, isBA=True, type="submap")
                if self.has_global:
                    # randomly select 10 keyframes from the global map
                    if len(self.global_keyframes) > 10:
                        random_keyframes = random.sample(self.global_keyframes, 10)
                        self.map(random_keyframes, self.gaussians, isBA=False, type="global")
                    else:
                        self.map(self.global_keyframes, self.gaussians, isBA=False, type="global")
                if self.last_sent >= 10:
                    self.map(self.current_window, self.sub_gaussians, noSplit=True, isBA=True, type="submap")
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    Log("Starting color refinement")
                    self.color_refinement()
                    Log("Color refinement done")
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf_submap(cur_frame_idx, viewpoint, depth_map=depth_map, init=True)
                    # self.initialize_map(cur_frame_idx, viewpoint)
                    self.initialize_submap(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")
                    
                elif data[0] == "end":
                    Log("Ending the system")
                    self.merge_submaps()
                    # the last 10 keyframes
                    self.map(self.global_keyframes[-10:], self.gaussians, iters=200, isBA=False, type="global")
                    self.push_to_frontend("end")
                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    # current_window = self.get_sub_window(current_window)

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    
                    # extand submap with new keyframe
                    self.add_next_kf_submap(cur_frame_idx, viewpoint, depth_map=depth_map)
                    
                    # check if want new submap
                    if self.should_start_new_submap(cur_frame_idx):
                        self.start_new_submap(cur_frame_idx, viewpoint)
                        self.map(self.current_window, self.sub_gaussians ,iters=self.iter_per_kf, isBA=False, type="submap")
                        self.map(self.current_window, self.sub_gaussians, noSplit=True, isBA=False, type="submap")
                    else:
                        opt_params = self.create_keyframe_optimizers(self.current_window)
                        if len(opt_params) > 0:
                            self.keyframe_optimizers = torch.optim.Adam(opt_params)
                            # local BA
                            self.map(self.current_window, self.sub_gaussians ,iters=self.iter_per_kf, isBA=True, type="submap")
                            self.map(self.current_window, self.sub_gaussians, noSplit=True, isBA=True, type="submap")
                        else:
                            self.map(self.current_window, self.sub_gaussians ,iters=self.iter_per_kf, isBA=False, type="submap")
                            self.map(self.current_window, self.sub_gaussians, noSplit=True, isBA=False, type="submap")
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return