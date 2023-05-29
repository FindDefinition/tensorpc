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

from tensorpc.flow.marker import mark_did_mount


class BufferMeshApp:

    @mark_create_layout
    def my_layout(self):
        import open3d as o3d

        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        mesh = o3d.io.read_triangle_mesh(
            "/root/tusimple/val_00800000_0.0001.ply")
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

        buffer_mesh = three.BufferMesh(
            buffers,
            mesh_points.shape[0],
            [
                three.MeshPhongMaterial().prop(color="aqua",
                                               specular="#ffffff",
                                               shininess=250,
                                               transparent=True),
                # three.MeshDepthMaterial(),

                # three.Edges(threshold=10, scale=1.1, color="black"),
            ]).prop(cast_shadow=True, receive_shadow=True)
        mesh_points = mesh_points[:500000]
        random_pcs = np.random.randint(-10, 10, size=[100, 3])
        random_pc_colors = np.random.uniform(0,
                                             255,
                                             size=[mesh_points.shape[0],
                                                   3]).astype(np.uint8)
        voxel_size = 0.1
        voxel_mesh = three.VoxelMesh(
            mesh_points.astype(np.float32),
            voxel_size,
            mesh_points.shape[0],
            [
                # three.MeshPhongMaterial().prop(vertex_colors=True, color="aqua", specular="#ffffff", shininess=250, transparent=True),
                three.MeshBasicMaterial().prop(vertex_colors=True),
                # three.Edges(),
            ],
            colors=random_pc_colors).prop()
        instanced_voxel_mesh = three.InstancedMesh(
            mesh_points.astype(np.float32),
            mesh_points.shape[0],
            [
                # three.MeshPhongMaterial().prop(vertex_colors=True, color="aqua", specular="#ffffff", shininess=250, transparent=True),
                three.BoxGeometry(voxel_size, voxel_size, voxel_size),
                three.MeshBasicMaterial().prop(),
            ],
            colors=random_pc_colors).prop()

        for k, v in buffers.items():
            print(k, v.shape)
        print(vert.shape, indices.shape)
        dirlight = three.DirectionalLight((64, 20, 15),
                                          target_position=(0, 20, 0),
                                          color=0xffffff,
                                          intensity=5).prop(cast_shadow=True)
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
        self.canvas = plus.SimpleCanvas(
            cam,
            init_canvas_childs=[
                # three.Mesh([
                #     three.BoxGeometry(),
                #     three.MeshStandardMaterial().prop(color="red"),
                #     # three.Edges(threshold=20, scale=1.1, color="black"),
                # ]).prop(position=(0, 0, 1), cast_shadow=True),
                # three.Mesh([
                #     three.PlaneGeometry(50, 50),
                #     three.MeshStandardMaterial().prop(color="#f0f0f0"),
                # ]).prop(receive_shadow=True, position=(0, 0, -0.1)),
                # three.AmbientLight(color=0xeeeeee, intensity=1),
                # dirlight,

                # dirlight.prop(helper_color=0x0f0f2a, helper_size=1.0),
                three.AmbientLight(),
                three.PointLight(color=0xffffff,
                                 intensity=5).prop(position=(13, 3, 5),
                                                   cast_shadow=True),
                # buffer_mesh,
                # voxel_mesh,
                instanced_voxel_mesh,
            ])
        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            mui.Button("Add", self._on_btn),
            self.canvas.prop(flex=1),
        ]).prop(min_height=0,
                min_width=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        return res

    async def _on_btn(self):
        pass


class BufferMeshDevApp:

    @mark_create_layout
    def my_layout(self):
        self.limit = 1000
        initial_num_pts = 500
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        random_pcs = np.random.randint(1, 20, size=[initial_num_pts, 3])
        random_pc_colors = np.random.uniform(0,
                                             255,
                                             size=[random_pcs.shape[0],
                                                   3]).astype(np.uint8)
        voxel_size = 0.1
        self.voxel_size = voxel_size
        voxel_mesh = three.VoxelMesh(
            random_pcs.astype(np.float32) * voxel_size,
            voxel_size,
            self.limit,
            [
                # three.MeshPhongMaterial().prop(vertex_colors=True, color="aqua", specular="#ffffff", shininess=250, transparent=False),
                three.MeshStandardMaterial().prop(vertex_colors=True),
                # three.MeshBasicMaterial().prop(vertex_colors=True),
                # three.Edges(),
                # three.Wireframe(),
            ],
            colors=random_pc_colors).prop(receive_shadow=True, cast_shadow=True)
        instanced_voxel_mesh = three.InstancedMesh(
            random_pcs.astype(np.float32) * voxel_size,
            random_pcs.shape[0],
            [
                # three.MeshPhongMaterial().prop(vertex_colors=True, color="aqua", specular="#ffffff", shininess=250, transparent=True),
                three.BoxGeometry(voxel_size, voxel_size, voxel_size),
                three.MeshStandardMaterial().prop(),
            ],
            colors=random_pc_colors).prop()
        self.voxel_mesh = voxel_mesh
        self.canvas = plus.SimpleCanvas(
            cam,
            init_canvas_childs=[
                # three.Environment().prop(preset="forest"),
                # three.PerformanceMonitor(),
                three.Sky().prop(sun_position=(1, 1, 1), distance=450000, inclination=0, azimuth=0.25),
                three.AmbientLight(),
                three.SpotLight((10, 10, 10)).prop(angle=0.25, penumbra=0.5, cast_shadow=True),
                # three.HemisphereLight(color=0xffffff, ground_color=0xb9b9b9, intensity=0.85).prop(position=(-7, 25, 13)),
                # three.PointLight(intensity=0.8).prop(position=(100, 100, 100),
                #                                    cast_shadow=True),
                # buffer_mesh,
                # voxel_mesh,
                voxel_mesh,
                three.Mesh([
                    three.PlaneGeometry(1000, 1000),
                    three.MeshStandardMaterial().prop(color="#f0f0f0"),
                ]).prop(receive_shadow=True, position=(0.0, 0.0, -0.1)),
                # three.Mesh([
                #     three.BoxGeometry(),
                #     three.MeshStandardMaterial().prop(color="orange"),
                # ]).prop(cast_shadow=True, position=(0, 5, 2)),
                # three.Mesh([
                #     three.BoxGeometry(),
                #     three.MeshStandardMaterial().prop(color="orange"),
                # ]).prop(cast_shadow=True, position=(0.45, 7, 1.25)),

            ])
        # <pointLight position={[100, 100, 100]} intensity={0.8} />
        # <hemisphereLight color="#ffffff" groundColor="#b9b9b9" position={[-7, 25, 13]} intensity={0.85} />

        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            mui.Button("750 Points", self._on_btn_750),
            mui.Button("250 Points", self._on_btn_250),
            self.canvas.prop(flex=1),
        ]).prop(min_height=0,
                min_width=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        return res

    async def _on_btn_750(self):
        pcs = np.random.randint(-10, 10, size=[75, 3])
        pc_colors = np.random.uniform(0, 255, size=[pcs.shape[0],
                                                    3]).astype(np.uint8)

        await self.canvas.send_and_wait(
            self.voxel_mesh.update_event(centers=pcs.astype(np.float32) *
                                         self.voxel_size,
                                         colors=pc_colors))

    async def _on_btn_250(self):
        pcs = np.random.randint(-10, 10, size=[25, 3])
        pc_colors = np.random.uniform(0, 255, size=[pcs.shape[0],
                                                    3]).astype(np.uint8)
        await self.canvas.send_and_wait(
            self.voxel_mesh.update_event(centers=pcs.astype(np.float32) *
                                         self.voxel_size,
                                         colors=pc_colors))

class BufferIndexedMeshApp:

    @mark_create_layout
    def my_layout(self):
        self.limit = 5000000
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        mesh = o3d.io.read_triangle_mesh(
            "/home/yy/Downloads/val_00800000_0.0001.ply")
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals).reshape(-1, 3).astype(np.float32)

        vertices = np.asarray(mesh.vertices).astype(np.float32)
        indices = np.asarray(mesh.triangles).reshape(-1).astype(np.int32)
        print(vertices.shape, indices.shape, normals.shape)
        
        # vertices = np.array([
        #     -1.0, -1.0,  1.0, 
        #     1.0, -1.0,  1.0, 
        #     1.0,  1.0,  1.0, 
        #     -1.0,  1.0,  1.0, 
        # ], np.float32).reshape(-1, 3)
        # indices = np.array([
        #     0, 1, 2,
        #     2, 3, 0,
        # ], np.int32)

        # vertices = np.array([
        #     -1.0, -1.0,  1.0, 
        #     1.0, -1.0,  1.0, 
        #     1.0,  1.0,  1.0, 

        #     1.0,  1.0,  1.0, 
        #     -1.0,  1.0,  1.0, 
        #     -1.0, -1.0,  1.0  
        # ], np.float32).reshape(-1, 3)
        buffer_mesh = three.BufferMesh({
                    "position": vertices,
                    # "normal": normals,
                }, self.limit, [
                    three.MeshPhongMaterial().prop(color="#f0f0f0"),
                ], initial_index=indices).prop(initial_calc_vertex_normals=True)
        self.buffer_mesh = buffer_mesh
        self.canvas = plus.SimpleCanvas(
            cam,
            init_canvas_childs=[
                three.Sky().prop(sun_position=(0, 1, 0), distance=450000, inclination=0, azimuth=0.25),
                three.AmbientLight(),
                three.SpotLight((10, 10, 5)).prop(angle=0.25, penumbra=0.5, cast_shadow=True),
                buffer_mesh,
            ])
        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            mui.Button("750 Points", self._on_btn_750),
            mui.Button("250 Points", self._on_btn_250),
            self.canvas.prop(flex=1),
        ]).prop(min_height=0,
                min_width=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        return res
    
    async def _on_btn_750(self):
        await self.buffer_mesh.calc_vertex_normals_in_frontend()

    async def _on_btn_250(self):
        pcs = np.random.randint(-10, 10, size=[25, 3])
        pc_colors = np.random.uniform(0, 255, size=[pcs.shape[0],
                                                    3]).astype(np.uint8)
        await self.canvas.send_and_wait(
            self.voxel_mesh.update_event(centers=pcs.astype(np.float32) *
                                         self.voxel_size,
                                         colors=pc_colors))

class MeshApp:

    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)

        self.canvas = plus.SimpleCanvas(
            cam,
            init_canvas_childs=[
                three.Mesh([
                    three.BoxGeometry(),
                    three.MeshStandardMaterial().prop(color="red"),
                    three.Edges(threshold=20, scale=1.1, color="black"),
                ]).prop(position=(0, 0, 1), cast_shadow=True),
                three.Mesh([
                    three.PlaneGeometry(50, 50),
                    three.MeshStandardMaterial().prop(color="#f0f0f0"),
                ]).prop(receive_shadow=True, position=(0, 0, -0.1)),
                three.PointLight(color=0xffffff,
                                 intensity=10).prop(position=(3, 3, 5),
                                                    cast_shadow=True),
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
