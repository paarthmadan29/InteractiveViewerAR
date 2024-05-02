import cv2
import numpy as np
import json

scene_image_via_prompt = False
prompt_choice = ["garden", "officeRoom", "luxuriousRoom"]
iter_choice = [0, 1]
if scene_image_via_prompt:
    # Load images
    scene_img = cv2.imread(f'sd_outputs/{prompt_choice[1]}/im_{iter_choice[0]}.png', cv2.IMREAD_COLOR)
    scene_img = cv2.resize(scene_img, (0, 0), fx=1.96875, fy=1.96875, interpolation=cv2.INTER_LINEAR)
else:
    # Load images
    scene_img = cv2.imread('data/scene.png', cv2.IMREAD_COLOR)
depth_img = cv2.imread('data/scene_depth.png', cv2.IMREAD_GRAYSCALE)

object_img = cv2.imread('data/object.png', cv2.IMREAD_UNCHANGED)
segmentation_img = cv2.imread('data/segmentation_image.png', cv2.IMREAD_GRAYSCALE)

# Load semantic mapping
with open('data/semantic_mapping.json') as f:
    semantic_mapping = json.load(f)

# Create a mask for valid placement locations (the floor in this case)
floor_mask = segmentation_img == semantic_mapping['floor'][0]
armchair_mask = segmentation_img == semantic_mapping['armchair'][0]
basket_mask = segmentation_img == semantic_mapping['basket'][0]
plaything_mask = segmentation_img == semantic_mapping['plaything'][1]

# Rescale object based on depth
def rescale_object(image, depth_value):
    print("Depth value:", depth_value)
    
    if depth_value > 175:
        scale = np.interp(depth_value, [175, 255], [0.9, 1.1])
    elif depth_value >= 30: # and depth_value <= 175:
        scale = np.interp(depth_value, [30, 175], [0.55, 0.9])
    else:
        scale = 0.55
    print(scale)
    
    scaled_height = int(image.shape[0] * scale)
    scaled_width = int(image.shape[1] * scale)
    resized_image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def rescale_shadow(image, scale_factor):
    shadow_alpha = 0.3  # Adjust the shadow transparency
    # Flip and scale the shadow to match the object size
    shadow_img = cv2.flip(image, 0)
    shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2BGRA)
    shadow_img[:, :, 3] = (shadow_img[:, :, 3] * shadow_alpha).astype(np.uint8)
    # Position the scaled shadow
    shadow_img_blurred = cv2.GaussianBlur(shadow_img, (21, 21), 0)

    # Reduce the opacity of the shadow further for a more subtle effect
    shadow_img_blurred[:, :, 3] = (shadow_img_blurred[:, :, 3] * 0.1).astype(np.uint8)

    shadow_img_reduced = cv2.resize(shadow_img, (0, 0), fx=1.0, fy=scale_factor, interpolation=cv2.INTER_AREA)
    shadow_img_reduced[:, :, :3] = 0  # Set color to black
    shadow_img_reduced[:, :, 3] = (shadow_img_reduced[:, :, 3] * 1.2).astype(np.uint8)  # Moderate opacity

    return shadow_img_reduced



# Overlay the object on the scene at the given position
def overlay_object(scene, object_img, position, depth_map, mask):
    x, y = position

    # if mask[y, x]:
    depth_value = depth_map[y, x] # depth value at a valid position
    resized_object_img = rescale_object(object_img, depth_value)

    resized_shadow_img = rescale_shadow(resized_object_img, scale_factor=0.4)

    # Calculate the region of interest
    # y1 = max(0, y - resized_object_img.shape[0] // 2 - resized_object_img.shape[0] // 2 + 120) 
    y1 = max(0, y - resized_object_img.shape[0] // 2 - resized_object_img.shape[0] // 2 + 120) 
    y2 = min(scene.shape[0], y + resized_object_img.shape[0] // 2 - resized_object_img.shape[0] // 2 + 120) 
    x1 = max(0, x - resized_object_img.shape[1] // 2)
    x2 = min(scene.shape[1], x + resized_object_img.shape[1] // 2)
    x1, y1, x2, y2 = int(np.ceil(x1)), int(np.ceil(y1)), int(np.ceil(x2)), int(np.ceil(y2))
    
    # cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    shadow_x1 = x1
    shadow_x2 = x2
    shadow_y1 = y2
    shadow_y2 = min(scene.shape[0], shadow_y1 + resized_shadow_img.shape[0])
    # cv2.rectangle(scene, (shadow_x1, shadow_y1), (shadow_x2, shadow_y2), (0, 0, 255), 3)
    # cv2.circle(scene, (shadow_x1, shadow_y1), 3, (255, 255, 0), 10, -1)
    # cv2.circle(scene, (shadow_x2, shadow_y2), 3, (0, 255, 0), 10, -1)
    alpha_s_shadow = resized_shadow_img[:, :, 3] / 255.0
    alpha_l_shadow = 1.0 - alpha_s_shadow
    if shadow_x2 <= scene.shape[1] and shadow_y2 <= scene.shape[0]:
        
        for c in range(0, 3):
            print("--------------------")
            print("alpha_s_shadow: ", alpha_s_shadow.shape)
            print("resized_shadow_img: " ,resized_shadow_img.shape)
            print("--------------------")
            scene[shadow_y1:shadow_y2, shadow_x1:shadow_x2, c] = (
                alpha_s_shadow[0:shadow_y2-shadow_y1, 0:shadow_x2-shadow_x1] * resized_shadow_img[0:shadow_y2-shadow_y1, 0:shadow_x2-shadow_x1, c] +
                alpha_l_shadow[0:shadow_y2-shadow_y1, 0:shadow_x2-shadow_x1] * scene[shadow_y1:shadow_y2, shadow_x1:shadow_x2, c]
            )
            print("shadow_applied")

    
    alpha_s = resized_object_img[:, :, 3] / 255.0
    alpha_l = 1 - alpha_s
    arm_mask_roi = armchair_mask[y1:y2, x1:x2]
    basket_mask_roi = basket_mask[y1:y2, x1:x2]
    plaything_mask_roi = plaything_mask[y1:y2, x1:x2]
    # Overlay the object
    
    does_intersect_arm = True in arm_mask_roi
    does_intersect_basket = True in basket_mask_roi
    does_intersect_plaything = True in plaything_mask_roi
    for c in range(3):
        
        if does_intersect_arm:
            if depth_value < 160:
                scene[y1:y2, x1:x2, c] = (np.where(1*arm_mask_roi == 1, alpha_s[0:y2-y1, 0:x2-x1]*scene[y1:y2, x1:x2, c], alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c])  + 
                                            alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])
            else:
                scene[y1:y2, x1:x2, c] = (alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c] +
                                    alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])
            # else:
                
        
        
        elif does_intersect_basket:
            if depth_value < 120:
                scene[y1:y2, x1:x2, c] = (np.where(1*basket_mask_roi == 1, alpha_s[0:y2-y1, 0:x2-x1]*scene[y1:y2, x1:x2, c], alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c])  + 
                                            alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])
            else:                
                scene[y1:y2, x1:x2, c] = (alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c] +
                                    alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])
                
        
        elif does_intersect_plaything:
            if depth_value < 30:
                scene[y1:y2, x1:x2, c] = (np.where(1*plaything_mask_roi == 1, alpha_s[0:y2-y1, 0:x2-x1]*scene[y1:y2, x1:x2, c], alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c])  + 
                                            alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])
        
            else:
                scene[y1:y2, x1:x2, c] = (alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c] +
                                    alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])
        else:
            scene[y1:y2, x1:x2, c] = (alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c] +
                                    alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])
    print("ARMCHAIR_INTERSECTION:", does_intersect_arm)
    print("BASKET_INTERSECTION:", does_intersect_basket)
    print("PLAYTHING_INTERSECTION:", does_intersect_plaything)

    # for c in range(3):
    #     scene[y1:y2, x1:x2, c] = (alpha_s[0:y2-y1, 0:x2-x1] * resized_object_img[0:y2-y1, 0:x2-x1, c] +
    #                                 alpha_l[0:y2-y1, 0:x2-x1] * scene[y1:y2, x1:x2, c])

    return scene

# Mouse callback function
def update_scene(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Clamp the mouse coordinates to the image size
        x = np.clip(x, 724, 3665) #depth_img.shape[1] - 1) # 
        y = np.clip(y, 0, 2903) #depth_img.shape[0] - 1) # 2288
        
        # Check if we are in a valid floor position
        if floor_mask[y, x] and not armchair_mask[y, x] and not basket_mask[y, x] and not plaything_mask[y, x]:
            # Now use the scalar depth_value to rescale and overlay the object
            print("(x, y) ", x, y)
             
            updated_scene = overlay_object(scene_img.copy(), object_img, (x, y), depth_img, floor_mask)
            # cv2.circle(updated_scene, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Interactive Viewer", updated_scene)


if __name__ == '__main__':
    # Set up the window and callback
    cv2.namedWindow("Interactive Viewer")
    cv2.setMouseCallback("Interactive Viewer", update_scene)

    # Show the initial scene
    cv2.imshow("Interactive Viewer", scene_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
