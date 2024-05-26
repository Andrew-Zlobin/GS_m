import torch
import numpy as np
import random
from scene import Scene
import torch.optim as optim
from os import makedirs
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.icomma_helper import load_LoFTR, get_pose_estimation_input
from utils.general_utils import print_stat
from utils.image_utils import to8b
import cv2
import imageio
import os
import ast
from scene.cameras import Camera_Pose
from utils.loss_utils import loss_loftr,loss_mse


rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])


# TODO: заменить на обычные, а то капец

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def trans_t_xyz(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

def draw_camera_in_top_camera(icomma_info, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, compute_grad_cov2d=True):
    
    # Пример матрицы start_pose_c2w
    start_pose_c2w = torch.tensor(viewpoint_camera, dtype=torch.float32).cuda()
    # np.array([
    #     [0.866, -0.5, 0, 1],
    #     [0.5, 0.866, 0, 2],
    #     [0, 0, 1, 3],
    #     [0, 0, 0, 1]
    # ])
    
    

    # Пример матрицы преобразования от мира к камере B (обратная матрица к start_pose_c2w)
    # Предположим, что это просто нихуя не тождественное преобразование
    world_to_cameraB = torch.tensor(np.linalg.inv(
        #np.eye(4)
        # rot_phi - поворот вокруг оптической оси камеры
        # rot_theta - поворот "налево"
        # rot_psi - поворот вверх-вниз
        trans_t_xyz(0,-10,0) @ rot_phi(0/180.*np.pi) @ rot_theta(0/180.*np.pi) @ rot_psi(-90/180.*np.pi)
        ), dtype=torch.float32).cuda()
    camera_pose = Camera_Pose(world_to_cameraB,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()
    # camera_pose для камеры с видом сверху
    camera_b_view = render(camera_pose,
                           gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)

    # Положение камеры в пространстве B
    # print(type(world_to_cameraB), type(start_pose_c2w))
    cameraB_pose = world_to_cameraB @ start_pose_c2w

    # Параметры проекции камеры B
    # Фокусное расстояние, координаты центра изображения, коэффициенты искажения и т. д.
    # Предположим, что они известны
    focal_length = 200 
    # ((camera_b_view.shape[1] / (2 * np.tan(icomma_info.FoVy / 2))) 
    #                 + 
    #                 (camera_b_view.shape[2] / (2 * np.tan(icomma_info.FoVx / 2)))) / 2
    image_center = torch.tensor(np.array([camera_b_view.shape[1] / 2, camera_b_view.shape[2] / 2]), dtype=torch.float32).cuda()  # Пример координат центра изображения
    distortion_coeffs = np.zeros(5)  # Пример коэффициентов искажения

    # Преобразование координат камеры в пространстве B в координаты на изображении
    # Это может быть выполнено с использованием функции проекции, например, функции cv2.projectPoints в OpenCV
    # Здесь просто приведен пример для наглядности
    camera_coordinates_B = cameraB_pose[:3, 3]
    image_coordinates_B = (focal_length * camera_coordinates_B[:2] / camera_coordinates_B[2]) + image_center

    # print("Координаты камеры на изображении с камеры B:", image_coordinates_B)
    # print("camera_b_view = ", camera_b_view.shape)
    rgb = camera_b_view.clone().permute(1, 2, 0).cpu().detach().numpy()
    rgb8 = to8b(rgb)
    filename = os.path.join('rendering.png')
    imageio.imwrite(filename, rgb8)
    cam_centre = image_coordinates_B.clone().cpu().detach().numpy()
    cam_centre = (int(cam_centre[0]), int(cam_centre[1]))
    # cv2.circle(to8b(camera_b_view.clone().cpu().detach().numpy()), cam_centre, 5, (0,255,0), thickness=1, lineType=8, shift=0)
    return to8b(camera_b_view.clone().cpu().detach().numpy()), cam_centre


                
def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path):
    # start pose & gt pose
    gt_pose_c2w=icomma_info.gt_pose_c2w
    start_pose_w2c=icomma_info.start_pose_w2c.cuda()
    camera_poses_sequence = []
    camera_b_view_query = []
    # query_image for comparing 
    query_image = icomma_info.query_image.cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()

    # store gif elements
    imgs=[]
    
    matching_flag= not icommaparams.deprecate_matching
    num_iter_matching = 0

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(),lr = icommaparams.camera_pose_lr)
    
    for k in range(icommaparams.pose_estimation_iter):

        rendering = render(camera_pose,
                           gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)#["render"]

        if matching_flag:
            loss_matching = loss_loftr(query_image,
                                       rendering,
                                       LoFTR_model,
                                       icommaparams.confidence_threshold_LoFTR,
                                       icommaparams.min_matching_points)
            
            loss_comparing = loss_mse(rendering,query_image)
            
            if loss_matching is None:
                loss = loss_comparing
            else:  
                loss = icommaparams.lambda_LoFTR *loss_matching + (1-icommaparams.lambda_LoFTR)*loss_comparing
                if loss_matching<0.001:
                    matching_flag=False
                    
            num_iter_matching += 1
        else:
            loss_comparing = loss_mse(rendering,query_image)
            loss = loss_comparing
            
            new_lrate = icommaparams.camera_pose_lr * (0.6 ** ((k - num_iter_matching + 1) / 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        
        # output intermediate results
        if (k + 1) % 10 == 0 or k == 0:
            print_stat(k, matching_flag, loss_matching, loss_comparing, 
                       camera_pose, gt_pose_c2w)
            # output images
            matrix_pose_c2w_to_top_camera = camera_pose.current_campose_c2w()
            a, b = draw_camera_in_top_camera(icomma_info, matrix_pose_c2w_to_top_camera, gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)
            a
            camera_poses_sequence.append(b)
            camera_b_view_query.append(a)
            print(a)
            print("current_campose_c2w = ", matrix_pose_c2w_to_top_camera)

            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    # ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k)+'.png')
                    #dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, rgb8)#, dst)
                    imgs.append(rgb8)#dst)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        camera_pose(start_pose_w2c)

    print(camera_poses_sequence)
    # camera_b_view = to8b(render(camera_pose,
    #                        gaussians, 
    #                        pipeline, 
    #                        background,
    #                        compute_grad_cov2d = icommaparams.compute_grad_cov2d).clone().permute(1, 2, 0).cpu().detach().numpy())
    for camera_poses_point in camera_poses_sequence:
        cv2.circle(camera_b_view_query[0], camera_poses_point, 5, (0,255,0), thickness=1, lineType=8, shift=0)
    imageio.imwrite('camera_path.png', rgb8)
    

    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=8)
        ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
        filename = os.path.join('ref.png')
        imageio.imwrite(filename, ref)

if __name__ == "__main__":

    args, model, pipeline, icommaparams = get_combined_args()

    makedirs(args.output_path, exist_ok=True)
    
    # load LoFTR_model
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


    LoFTR_model=load_LoFTR(icommaparams.LoFTR_ckpt_path,icommaparams.LoFTR_temp_bug_fix)
    
    # load gaussians
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get camera info from Scene
    # Reused 3DGS code to obtain camera information. 
    # You can customize the iComMa_input_info in practical applications.
    scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False)
    obs_view=scene.getTestCameras()[args.obs_img_index]
    #obs_view=scene.getTrainCameras()[args.obs_img_index]
    icomma_info=get_pose_estimation_input(obs_view,ast.literal_eval(args.delta))
    
    # pose estimation
    camera_pose_estimation(gaussians,background,pipeline,icommaparams,icomma_info,args.output_path)