# InteractiveViewerAR

This repository contains the code for an Interactive Viewer which helps visualize 2D objects in 2D scenarios.
It used primarily two libraries:
- numpy, for array operations
- openCV for image operations and visualization

#### Given the depth map and segmentation map of the backdrop scene, the aim is to allow the user to traverse the foreground object (here, a chair) around the scene, making sure of of the following few things

1. Object is allowed to traverse only on the valid floor region
2. Object should rescale automatically as it traverses towards and away from the POV
3. Object should not move into the other objects on the floor and avoid occlusion

### Proposed Approach:
1. Create a MOUSE_MOVE event callback function that updates the scene whenever mouse coordinates update
2. A boolen check for valid traverse path using the "floor" mapping in the segmentation given. (segmentation_image.png and segmentation_mapping.json)
3. An overlay function that utilizes the alpha transparency map of the object (4th channel) to overlay the foreground object over the backdrop scene based on the region of interest selected (ROI).
4. Utilize the depth value at every valid point to find the estimated size of the object based on some heuristic.
5. Ultilize depth value of the object at the current location compared to the depth value of the obstacle on the floor to judge if occlusion occurs and overlay accordingly.
6. Create a flipped, gaussian blur mask of the rescaled object to have a shadow effect that moves along the foreground object.


### [BONUS]
To create a modified scene with the same depth map using Stable Diffusion and controlNet Depth model using a text prompt

Using sd_generate.py, we use huggingface diffusers and 2 pretrained_models:
1. <a href="https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE">SG161222/Realistic_Vision_V6.0_B1_noVAE</a>: For txt2img Pipeline
2. <a href="https://huggingface.co/lllyasviel/sd-controlnet-depth/tree/main">lllyasviel/sd-controlnet-depth</a>: for ControlNet Depth Pipeline

We re-apply the modified scene in the script to see the outputs.

Some sample outputs are <a href="https://www.google.com/">here</a>
