import os

from nuscenes.nuscenes import NuScenes

project_folder = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(project_folder, "v1.0-trainval08_blobs")
version = "v1.0-trainval"
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

camera_name = "CAM_FRONT"
max_images = 1000

nusc.list_scenes()

# # Output folder
# output_folder = os.path.join(project_folder, "output")
# os.makedirs(output_folder, exist_ok=True)
#
# count = 0
# for sample in nusc.sample:
#     if count >= max_images:
#         break
#
#     cam_token = sample['data'][camera_name]
#     cam_data = nusc.get('sample_data', cam_token)
#     img_path = os.path.join(nusc.dataroot, cam_data['filename'])
#     if not os.path.exists(img_path):
#         continue
#
#     img = np.array(Image.open(img_path))
#
#     # Camera calibration
#     cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
#     cam_intrinsic = np.array(cs['camera_intrinsic'])
#     sensor_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
#
#     fig, ax = plt.subplots(1, figsize=(12, 8))
#     ax.imshow(img)
#
#     for ann_token in sample['anns']:
#         ann = nusc.get('sample_annotation', ann_token)
#         if int(ann['visibility_token']) >= 2:
#             # Create 3D box
#             box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
#
#             # Move box to ego vehicle frame
#             box.translate(-np.array(sensor_pose['translation']))
#             box.rotate(Quaternion(sensor_pose['rotation']).inverse)
#
#             # Move box to camera frame
#             box.translate(-np.array(cs['translation']))
#             box.rotate(Quaternion(cs['rotation']).inverse)
#
#             # Project 3D box corners to 2D image
#             corners = view_points(box.corners(), cam_intrinsic, normalize=True)  # shape (3,8)
#
#             # Only keep boxes that are in front of the camera
#             if np.any(corners[2, :] < 0.1):
#                 continue
#
#             u = corners[0, :]
#             v = corners[1, :]
#             xmin, xmax = int(u.min()), int(u.max())
#             ymin, ymax = int(v.min()), int(v.max())
#
#             # Clip coordinates to image size
#             h, w = img.shape[:2]
#             xmin, xmax = np.clip([xmin, xmax], 0, w - 1)
#             ymin, ymax = np.clip([ymin, ymax], 0, h - 1)
#
#             # Draw rectangle
#             ax.add_patch(plt.Rectangle(
#                 (xmin, ymin),
#                 xmax - xmin,
#                 ymax - ymin,
#                 fill=False,
#                 edgecolor='red',
#                 linewidth=2
#             ))
#
#     # Save annotated image
#     save_path = os.path.join(output_folder, f"sample_{count}_{sample['token']}.png")
#     plt.savefig(save_path)
#     plt.close(fig)
#
#     count += 1
