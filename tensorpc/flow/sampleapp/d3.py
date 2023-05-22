# Copyright 2023 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorpc.flow import mui, three, plus, mark_create_layout, appctx
import sys
from tensorpc import PACKAGE_ROOT
import numpy as np 
import scipy.spatial
import open3d as o3d 

class BufferMeshApp:
    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        mesh = o3d.io.read_triangle_mesh("/root/tusimple/val_00800000_0.0001.ply")
        mesh.compute_vertex_normals()

        vert = np.asarray(mesh.vertices)
        indices = np.asarray(mesh.triangles)
        mesh_points = vert[indices.reshape(-1)].reshape(-1, 3)
        normals = np.asarray(mesh.triangle_normals).reshape(-1, 1, 3)
        normals = np.tile(normals, (1, 3, 1)).reshape(-1, 3)
        colors = np.zeros_like(mesh_points).astype(np.uint8)
        print(mesh_points.shape)
        mesh_points = np.ascontiguousarray(mesh_points)
        buffers = {
            "position": mesh_points.astype(np.float32),
            "normal": normals.astype(np.float32),
            "color": colors.astype(np.float32),
        }

        buffer_mesh = three.BufferMesh(buffers, mesh_points.shape[0], [
            three.MeshPhongMaterial().prop(color="aqua", specular="#ffffff", shininess=250, transparent=True),
            # three.MeshDepthMaterial(),

            # three.Edges(threshold=10, scale=1.1, color="black"),
        ]).prop(cast_shadow=True, receive_shadow=True)
        mesh_points = mesh_points[:500000]
        random_pcs = np.random.randint(-10, 10, size=[100, 3])
        random_pc_colors = np.random.uniform(0, 255, size=[mesh_points.shape[0], 3]).astype(np.uint8)
        voxel_size = 0.1
        voxel_mesh = three.VoxelMesh(mesh_points.astype(np.float32), voxel_size, mesh_points.shape[0], [
            # three.MeshPhongMaterial().prop(vertex_colors=True, color="aqua", specular="#ffffff", shininess=250, transparent=True),
            three.MeshBasicMaterial().prop(vertex_colors=True),
        ], colors=random_pc_colors).prop(cast_shadow=True, receive_shadow=True)
        for k, v in buffers.items():
            print(k, v.shape)
        print(vert.shape, indices.shape)
        dirlight = three.DirectionalLight((64, 20, 15), target_position=(0, 20, 0), color=0xffffff, intensity=5).prop(cast_shadow=True)
        dirlight.set_sx_props({
            "shadow-mapSize-height": 2048,
            "shadow-mapSize-width": 2048,
            # "shadow-camera-near": 0.5,
            # "shadow-camera-far": 1000,
            # "shadow-camera-left": -1000,
            # "shadow-camera-right": 1000,
            # "shadow-camera-top": 1000,
            # "shadow-camera-bottom": -1000,

        })
        self.canvas = plus.SimpleCanvas(cam, init_canvas_childs=[
            # three.PrimitiveMesh([
            #     three.BoxGeometry(), 
            #     three.MeshStandardMaterial().prop(color="red"),
            #     # three.Edges(threshold=20, scale=1.1, color="black"),
            # ]).prop(position=(0, 0, 1), cast_shadow=True),
            # three.PrimitiveMesh([
            #     three.PlaneGeometry(50, 50),
            #     three.MeshStandardMaterial().prop(color="#f0f0f0"),
            # ]).prop(receive_shadow=True, position=(0, 0, -0.1)),
            # three.AmbientLight(color=0xeeeeee, intensity=1),
            # dirlight,

            # dirlight.prop(helper_color=0x0f0f2a, helper_size=1.0),
            three.AmbientLight(),
            three.PointLight(color=0xffffff, intensity=5).prop(position=(13, 3, 5), cast_shadow=True),
            # buffer_mesh,
            voxel_mesh,
        ])
        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            self.canvas.prop(flex=1),
        ]).prop(min_height=0,
                min_width=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        return res



class MeshApp:
    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)

        self.canvas = plus.SimpleCanvas(cam, init_canvas_childs=[
            three.PrimitiveMesh([
                three.BoxGeometry(), 
                three.MeshStandardMaterial().prop(color="red"),
                three.Edges(threshold=20, scale=1.1, color="black"),
            ]).prop(position=(0, 0, 1), cast_shadow=True),
            three.PrimitiveMesh([
                three.PlaneGeometry(50, 50),
                three.MeshStandardMaterial().prop(color="#f0f0f0"),
            ]).prop(receive_shadow=True, position=(0, 0, -0.1)),
            three.PointLight(color=0xffffff, intensity=10).prop(position=(3, 3, 5), cast_shadow=True),
        ])
        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            self.canvas.prop(flex=1),
        ]).prop(min_height=0,
                min_width=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        return res


class CollectionApp:
    @mark_create_layout
    def my_layout(self):
        appctx.get_app().set_enable_language_server(True)
        pyright_setting = appctx.get_app().get_language_server_settings()
        pyright_setting.python.analysis.pythonPath = sys.executable
        pyright_setting.python.analysis.extraPaths = [
            str(PACKAGE_ROOT.parent),
        ]
        return plus.InspectPanel(self)

