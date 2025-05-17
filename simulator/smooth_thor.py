import ai2thor.build
from ai2thor._quality_settings import DEFAULT_QUALITY
from ai2thor.controller import Controller
import numpy as np
import copy
from PIL import Image
import random
from copy import deepcopy
from typing import Any, Dict, Optional
import ai2thor
from ai2thor.server import DepthFormat, Event
from ai2thor._quality_settings import DEFAULT_QUALITY


class Smooth_Thor(Controller):
    def __init__(
        self,
        quality=DEFAULT_QUALITY,
        fullscreen=False,
        headless=False,
        port=0,
        start_unity=True,
        local_executable_path=None,
        local_build=False,
        commit_id=ai2thor.build.COMMIT_ID,
        branch=None,
        width=300,
        height=300,
        x_display=None,
        host="127.0.0.1",
        scene=None,
        image_dir=".",
        save_image_per_frame=False,
        depth_format=DepthFormat.Meters,
        add_depth_noise=False,
        download_only=False,
        include_private_scenes=False,
        server_class=None,
        gpu_device=None,
        platform=None,
        server_timeout: Optional[float] = 100.0,
        server_start_timeout: float = 300.0,
        **unity_initialization_parameters,
    ):
        self.reset_values(unity_initialization_parameters)
        super().__init__(quality=quality, 
                         fullscreen=fullscreen, 
                         headless=headless, 
                         port=port, 
                         start_unity=start_unity, 
                         local_executable_path=local_executable_path, 
                         local_build=local_build, 
                         commit_id=commit_id, 
                         branch=branch, 
                         width=width, 
                         height=height, 
                         x_display=x_display, 
                         host=host, 
                         scene=scene, 
                         image_dir=image_dir, 
                         save_image_per_frame=save_image_per_frame, 
                         depth_format=depth_format, 
                         add_depth_noise=add_depth_noise, 
                         download_only=download_only, 
                         include_private_scenes=include_private_scenes, 
                         server_class=server_class, 
                         gpu_device=gpu_device, 
                         platform=platform, 
                         server_timeout=server_timeout, 
                         server_start_timeout=server_start_timeout, 
                         **unity_initialization_parameters)
        self.object_class2color = self.init_object_class2color()
    
    def reset_values(self, args: Dict[str, Any] = None):
        # check if the smooth_factor in args, if not, check if the smooth_factor in self, else set to default value 2
        self.smooth_factor = args.get("smooth_factor", self.__getattribute__("smooth_factor") if hasattr(self, "smooth_factor") else 2)
        self.smooth_nav = args.get("smooth_nav", self.__getattribute__("smooth_nav") if hasattr(self, "smooth_nav") else False)
        self.record_video = args.get("record_video", self.__getattribute__("record_video") if hasattr(self, "record_video") else True)
        self.gridSize = args.get("gridSize", self.__getattribute__("gridSize") if hasattr(self, "gridSize") else 0.25)
        self.rotateStepDegrees = args.get("rotateStepDegrees", self.__getattribute__("rotateStepDegrees") if hasattr(self, "rotateStepDegrees") else 90)
        self.snapToGrid = args.get("snapToGrid", self.__getattribute__("snapToGrid") if hasattr(self, "snapToGrid") else True)
        if not hasattr(self, "images"):
            self.images = []
        else:
            self.images.clear()
        args.pop("smooth_factor", None)
        args.pop("smooth_nav", None)
        args.pop("record_video", None)
        assert not (self.smooth_nav and self.snapToGrid), "snapToGrid must be False when smooth nav is True"

    def init_object_class2color(self):
        object_class2color = {}
        colors = self.last_event.metadata['colors']
        if colors is None:
            return object_class2color
        object_type_list = [object["objectType"] for object in self.last_event.metadata['objects']]
        object_type_list = list(set(object_type_list))
        for color_data in colors:
            color = color_data['color']
            name: str = color_data['name']
            if name in object_type_list:
                object_class2color[name.lower()] = color
        return object_class2color
    
    def reset(self, scene=None, **init_params):
        self.reset_values(init_params)
        event = super().reset(scene, **init_params)
        self.object_class2color = self.init_object_class2color()
        return event
    
    def save_image(self, event: Event):
        if not self.record_video:
            return
        image = {
            "rgb": event.frame,
            "depth": event.depth_frame,
            "class_segmentation": event.class_segmentation_frame,
            "instance_segmentation": event.instance_segmentation_frame,
            "third_party_camera_frames": event.third_party_camera_frames,
        }
        self.images.append(image)

    def detect_class_in_image(self, target_object_class: str, class_image: np.ndarray) -> int:
        target_color = self.object_class2color[target_object_class.lower()]
        # check if the target color in the class image
        return sum(np.all(class_image == target_color, axis=-1))
    
    def add_map_view_camera(self) -> Event:
        # Setup the top-down camera
        event = super().step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = super().step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        return event
        
    def step(self, action, **action_args):
        if self.smooth_nav:
            event = self.smooth_step(action, **action_args)
        else:
            event = super().step(action, **action_args)
        self.save_image(event)
        return event
        
    def smooth_step(self, action, **action_args):
        action_type = action['action'] if isinstance(action, dict) else action
        if "Move" in action_type:
            if isinstance(action, dict):
                moveMagnitude = action.get("moveMagnitude", None)
            else:
                moveMagnitude = action_args.get("moveMagnitude", None)
            event = self.smooth_move_action(action_type, moveMagnitude=moveMagnitude)
        elif "Rotate" in action_type:
            if isinstance(action, dict):
                degrees = action.get("degrees", None)
            else:
                degrees = action_args.get("degrees", None)
            event = self.smooth_rotate(action_type, degrees=degrees)
        elif "Look" in action_type:
            if isinstance(action, dict):
                degrees = action.get("degrees", None)
            else:
                degrees = action_args.get("degrees", None)
            event = self.smooth_look(action_type, degrees=degrees)
        else:
            event = super().step(action, **action_args)
        return event

    def smooth_move_action(self, action_type: str, moveMagnitude: float = None):
        if moveMagnitude is None:
            moveMagnitude = self.gridSize
        moveMagnitude = moveMagnitude / self.smooth_factor

        for _ in range(self.smooth_factor-1):
            event = super().step(action_type, moveMagnitude=moveMagnitude)
            self.save_image(event)

        event = super().step(action_type, moveMagnitude=moveMagnitude)
        return event
    
    def smooth_rotate(self, action_type: str, degrees: float = None):
        '''
        smoother RotateLeft and RotateRight
        '''
        degrees = degrees or self.rotateStepDegrees
        assert degrees > 0
        rotation = self.last_event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action_type == 'RotateLeft':
            end_rotation = (start_rotation - degrees)
        else:
            end_rotation = (start_rotation + degrees)

        for xx in np.arange(.1, 1, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'Teleport',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'Teleport',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                }
                event = super().step(teleport_action)
            self.save_image(event)
            
        event = super().step(action="Teleport", rotation=end_rotation)
        return event

    def smooth_look(self, action_type: str, degrees: float = None):
        '''
        smoother LookUp and LookDown
        '''
        degrees = degrees or self.rotateStepDegrees
        assert degrees > 0
        horizon = self.last_event.metadata['agent']['cameraHorizon']
        start_horizon = horizon
        if action_type == 'LookUp':
            end_horizon = (start_horizon + degrees)
        else:
            end_horizon = (start_horizon - degrees)

        for xx in np.arange(.1, 1, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'Teleport',
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'Teleport',
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)
            self.save_image(event)
            
        event = super().step(action="Teleport", horizon=end_horizon)
        return event
    
