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

import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from tensorpc.flow import mui, three, plus, mark_create_layout, appctx
import sys
from tensorpc import PACKAGE_ROOT
import numpy as np

from tensorpc.flow.marker import mark_did_mount
from tensorpc import prim


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
            ]).prop(castShadow=True, receiveShadow=True)
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
                # three.MeshPhongMaterial().prop(vertexColors=True, color="aqua", specular="#ffffff", shininess=250, transparent=True),
                three.MeshBasicMaterial().prop(vertexColors=True),
                # three.Edges(),
            ],
            colors=random_pc_colors).prop()
        instanced_voxel_mesh = three.InstancedMesh(
            mesh_points.astype(np.float32),
            mesh_points.shape[0],
            [
                # three.MeshPhongMaterial().prop(vertexColors=True, color="aqua", specular="#ffffff", shininess=250, transparent=True),
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
                                          intensity=5).prop(castShadow=True)
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
                # ]).prop(position=(0, 0, 1), castShadow=True),
                # three.Mesh([
                #     three.PlaneGeometry(50, 50),
                #     three.MeshStandardMaterial().prop(color="#f0f0f0"),
                # ]).prop(receiveShadow=True, position=(0, 0, -0.1)),
                # three.AmbientLight(color=0xeeeeee, intensity=1),
                # dirlight,

                # dirlight.prop(helper_color=0x0f0f2a, helper_size=1.0),
                three.AmbientLight(),
                three.PointLight(color=0xffffff,
                                 intensity=5).prop(position=(13, 3, 5),
                                                   castShadow=True),
                # buffer_mesh,
                # voxel_mesh,
                instanced_voxel_mesh,
            ])
        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            mui.Button("Add", self._on_btn),
            self.canvas.prop(flex=1),
        ]).prop(minHeight=0,
                minWidth=0,
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
                # three.MeshPhongMaterial().prop(vertexColors=True, color="aqua", specular="#ffffff", shininess=250, transparent=False),
                three.MeshStandardMaterial().prop(vertexColors=True),
                # three.MeshBasicMaterial().prop(vertexColors=True),
                # three.Edges(),
                # three.Wireframe(),
            ],
            colors=random_pc_colors).prop(receiveShadow=True, castShadow=True)
        instanced_voxel_mesh = three.InstancedMesh(
            random_pcs.astype(np.float32) * voxel_size,
            random_pcs.shape[0],
            [
                # three.MeshPhongMaterial().prop(vertexColors=True, color="aqua", specular="#ffffff", shininess=250, transparent=True),
                three.BoxGeometry(voxel_size, voxel_size, voxel_size),
                three.MeshStandardMaterial().prop(),
            ],
            colors=random_pc_colors).prop()
        self.voxel_mesh = voxel_mesh
        self.canvas = plus.SimpleCanvas(
            cam,
            init_canvas_childs=[
                # three.GLTFLoaderContext("tensorpc://porsche-transformed.glb", [

                # ]),
                # three.Environment
                three.Environment([
                    three.AmbientLight(),
                ]).prop(files="tensorpc://old_depot_2k.hdr",
                        ground=three.EnvGround(radius=130, height=32)),
                # three.PerformanceMonitor(),
                three.Sky().prop(sunPosition=(1, 1, 1),
                                 distance=450000,
                                 inclination=0,
                                 azimuth=0.25),
                three.AmbientLight(),
                three.SpotLight((10, 10, 10)).prop(angle=0.25,
                                                   penumbra=0.5,
                                                   castShadow=True),
                # three.HemisphereLight(color=0xffffff, ground_color=0xb9b9b9, intensity=0.85).prop(position=(-7, 25, 13)),
                # three.PointLight(intensity=0.8).prop(position=(100, 100, 100),
                #                                    castShadow=True),
                # buffer_mesh,
                # voxel_mesh,
                voxel_mesh,
                three.Mesh([
                    three.PlaneGeometry(1000, 1000),
                    three.MeshStandardMaterial().prop(color="#f0f0f0"),
                ]).prop(receiveShadow=True, position=(0.0, 0.0, -0.1)),
                # three.Mesh([
                #     three.BoxGeometry(),
                #     three.MeshStandardMaterial().prop(color="orange"),
                # ]).prop(castShadow=True, position=(0, 5, 2)),
                # three.Mesh([
                #     three.BoxGeometry(),
                #     three.MeshStandardMaterial().prop(color="orange"),
                # ]).prop(castShadow=True, position=(0.45, 7, 1.25)),
            ])
        # <pointLight position={[100, 100, 100]} intensity={0.8} />
        # <hemisphereLight color="#ffffff" groundColor="#b9b9b9" position={[-7, 25, 13]} intensity={0.85} />
        appctx.get_app().add_file_resource("porsche-transformed.glb",
                                           self.porsche)
        appctx.get_app().add_file_resource("old_depot_2k.hdr",
                                           self.old_depot_2k)
        appctx.get_app().add_file_resource("std.png", self.std_png)

        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            mui.Button("750 Points", self._on_btn_750),
            mui.Button("250 Points", self._on_btn_250),
            self.canvas.prop(flex=1),
        ]).prop(minHeight=0,
                minWidth=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        return res

    def std_png(self):
        return mui.FileResource(
            name="std.png",
            path=str(Path.home() /
                     "Pictures/Screenshot from 2023-03-10 15-40-39.png"),
            content_type="image/png")

    def old_depot_2k(self):
        return mui.FileResource(name="old_depot_2k.hdr",
                                path=str(Path.home() / "old_depot_2k.hdr"))

    def porsche(self):
        return mui.FileResource(name="porsche-transformed.glb",
                                path=str(Path.home() /
                                         "porsche-transformed.glb"))

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


async def download_file(url: str, chunk_size: int = 2**16):
    sess = prim.get_http_client_session()
    chunks = []
    async with sess.get(url) as response:
        assert response.status == 200
        while True:
            chunk = await response.content.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


class EnvmapGroupdProjectionApp:

    def __init__(self) -> None:
        self.hdr_content: Optional[bytes] = None
        self.glb_content: Optional[bytes] = None

    @mark_create_layout
    def my_layout(self):
        self.limit = 1000
        initial_num_pts = 500
        cam = three.PerspectiveCamera(fov=35,
                                      up=(0, 1, 0)).prop(position=(-30, 100,
                                                                   120))
        car_group = three.Group([
            three.CubeCamera([
                three.Group([
                    three.Group([
                        three.Mesh([]).prop(
                            position=(-7.966238, -0.10155, -7.966238),
                            scale=0.000973).set_override_props_unchecked(
                                geometry="nodes.mesh_1_instance_0.geometry",
                                material="materials.930_plastics"),
                        three.Mesh([]).prop(
                            position=(-7.966238, -0.10155, -7.966238),
                            scale=0.000973).set_override_props_unchecked(
                                geometry="nodes.mesh_1_instance_1.geometry",
                                material="materials.930_plastics"),
                    ]).prop(rotation=(np.pi / 2, 0, 0)),
                    three.Group([
                        three.SelectionContext([
                            three.EffectComposer([
                                three.Outline().prop(blur=True, edgeStrength=100, 
                                                     width=1000, visibleEdgeColor=0xf43f12, 
                                                     hiddenEdgeColor=0xf43f12),
                            ]).prop(autoClear=False, multisampling=0),
                            three.Mesh([
                            ]).set_override_props_unchecked_dict({
                                "geometry":
                                "nodes.mesh_0.geometry",
                                "material":
                                "materials.paint",
                                "material-envMap":
                                "CubeCameraTexture",
                            }).set_sx_props({
                                "material-color": "#ffdf71",
                            }).prop(enableSelect=True, selectOverrideProps={
                                "material-color": "#aadf71",
                            }, userData={
                                "RTX": "4090Ti"
                            }),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_1.geometry",
                                material="materials.930_chromes"),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_2.geometry",
                                material="materials.black"),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_3.geometry",
                                material="materials.930_lights"),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_4.geometry",
                                material="materials.glass"),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_5.geometry",
                                material="materials.930_stickers"),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_6.geometry",
                                material="materials.930_plastics").set_sx_props({
                                    "material-polygonOffset":
                                    True,
                                    "material-polygonOffsetFactor":
                                    -10,
                                }),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_7.geometry",
                                material="materials.930_lights_refraction"),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_8.geometry",
                                material="materials.930_rim"),
                            three.Mesh([]).set_override_props_unchecked(
                                geometry="nodes.mesh_0_9.geometry",
                                material="materials.930_tire"),

                            ], lambda x: print(x)),

                    ]).prop(position=(-7.966238, -0.10155, -7.966238),
                            scale=0.000973),
                ]).prop(position=(0, -1.5, 0)),
            ]).prop(frames=1, position=(0.0, 1.5, 0), near=0.1,
                    resolution=128),
            three.Group([
                three.Mesh([]).set_override_props_unchecked(
                    geometry="nodes.mesh_2.geometry",
                    material="materials.plate").set_sx_props({
                        "material-roughness":
                        1,
                    }),
                three.Mesh([]).set_override_props_unchecked(
                    geometry="nodes.mesh_2_1.geometry",
                    material="materials.DefaultMaterial"),
                three.Mesh([]).set_override_props_unchecked(
                    geometry="nodes.mesh_2_2.geometry",
                    material=r"materials.Material\.001").set_sx_props({
                        "material-depthWrite":
                        False,
                        "material-opacity":
                        0.6,
                    }),
            ]).prop(position=(-7.966238, -0.10155, -7.966238), scale=0.000973),
        ])
        car = three.URILoaderContext(
            three.URILoaderType.GLTF, "tensorpc://porsche-transformed.glb", [
                car_group.prop(position=(-8, 0, -2), scale=20).set_sx_props(
                    {"rotation-y": -np.pi / 4}),
            ])
        self.canvas = plus.SimpleCanvas(
            cam,
            init_canvas_childs=[
                # three.Environment
                three.AmbientLight(),
                three.Environment([]).prop(files="tensorpc://old_depot_2k.hdr",
                                           ground=three.EnvGround(radius=130,
                                                                  height=32)),
                car,
                three.SpotLight((-80, 200, -100)).prop(angle=1, intensity=1),
                three.ContactShadows().prop(renderOrder=2,
                                            frames=1,
                                            resolution=1024,
                                            scale=120,
                                            blur=2,
                                            opacity=0.6,
                                            far=100)
                # three.PerformanceMonitor(),
            ])
        # <pointLight position={[100, 100, 100]} intensity={0.8} />
        # <hemisphereLight color="#ffffff" groundColor="#b9b9b9" position={[-7, 25, 13]} intensity={0.85} />
        appctx.get_app().add_file_resource("porsche-transformed.glb",
                                           self.porsche)
        appctx.get_app().add_file_resource("old_depot_2k.hdr",
                                           self.old_depot_2k)

        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            mui.Button("dev_rotate", lambda : self.canvas.ctrl.rotate_to(0, 1.57)),
            self.canvas.prop(flex=1),
        ]).prop(minHeight=0,
                minWidth=0,
                flex=1,
                width="100%",
                height="100%",
                overflow="hidden")
        return res

    async def old_depot_2k(self):
        url = "https://uploads.codesandbox.io/uploads/user/b3e56831-8b98-4fee-b941-0e27f39883ab/KNRT-old_depot_2k.hdr"
        if self.hdr_content is None:
            self.hdr_content = await download_file(url)
        print("self.hdr_content", len(self.hdr_content))
        return mui.FileResource(name="old_depot_2k.hdr",
                                content=self.hdr_content)

    async def porsche(self):
        url = "https://uploads.codesandbox.io/uploads/user/b3e56831-8b98-4fee-b941-0e27f39883ab/or72-porsche-transformed.glb"
        url2 = "https://uploads.codesandbox.io/uploads/user/b3e56831-8b98-4fee-b941-0e27f39883ab/cExH-911-transformed.glb"
        if self.glb_content is None:
            self.glb_content = await download_file(url)
        print("self.glb_content", len(self.glb_content))

        return mui.FileResource(name="porsche-transformed.glb",
                                content=self.glb_content)


class BufferIndexedMeshApp:

    @mark_create_layout
    def my_layout(self):
        self.limit = 5000000
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000)
        mesh = o3d.io.read_triangle_mesh(
            "/home/yy/Downloads/val_00800000_0.0001.ply")
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals).reshape(-1,
                                                          3).astype(np.float32)

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
        buffer_mesh = three.BufferMesh(
            {
                "position": vertices,
                # "normal": normals,
            },
            self.limit,
            [
                three.MeshPhongMaterial().prop(color="#f0f0f0"),
            ],
            initial_index=indices).prop(initialCalcVertexNormals=True)
        self.buffer_mesh = buffer_mesh
        self.canvas = plus.SimpleCanvas(cam,
                                        init_canvas_childs=[
                                            three.Sky().prop(sunPosition=(0, 1,
                                                                          0),
                                                             distance=450000,
                                                             inclination=0,
                                                             azimuth=0.25),
                                            three.AmbientLight(),
                                            three.SpotLight(
                                                (10, 10,
                                                 5)).prop(angle=0.25,
                                                          penumbra=0.5,
                                                          castShadow=True),
                                            buffer_mesh,
                                        ])
        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            mui.Button("750 Points", self._on_btn_750),
            mui.Button("250 Points", self._on_btn_250),
            self.canvas.prop(flex=1),
        ]).prop(minHeight=0,
                minWidth=0,
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
                ]).prop(position=(0, 0, 1), castShadow=True),
                three.Mesh([
                    three.PlaneGeometry(50, 50),
                    three.MeshStandardMaterial().prop(color="#f0f0f0"),
                ]).prop(receiveShadow=True, position=(0, 0, -0.1)),
                three.PointLight(color=0xffffff,
                                 intensity=10).prop(position=(3, 3, 5),
                                                    castShadow=True),
            ])
        self.canvas.canvas.prop(shadows=True)
        res = mui.VBox([
            self.canvas.prop(flex=1),
        ]).prop(minHeight=0,
                minWidth=0,
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


async def download_file_Dev(url: str, chunk_size: int = 2**16):
    sess = aiohttp.ClientSession()
    chunks = []
    async with sess.get(url) as response:
        assert response.status == 200
        while True:
            chunk = await response.content.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)

class App:
    @mark_create_layout
    def my_layout(self):
        cam = three.PerspectiveCamera(fov=75, near=0.1, far=1000).prop(position=(0, 0, 5))
        canvas = plus.SimpleCanvas(init_canvas_childs=[
            cam,
            three.CameraControl().prop(makeDefault=True),
            three.AmbientLight(),
            three.PointLight().prop(position=(13, 3, 5),
                                    castShadow=True,
                                    color=0xffffff,
                                    intensity=5),
            three.Mesh([
                three.PlaneGeometry(1000, 1000),
                three.MeshStandardMaterial().prop(color="#f0f0f0"),
            ]).prop(receiveShadow=True, position=(0.0, 0.0, -2)),

            # three.SelectionContext([
            #                 three.EffectComposer([
            #                     three.Outline().prop(blur=True, edgeStrength=100, 
            #                                          width=1000, visibleEdgeColor=0xfff, 
            #                                          hiddenEdgeColor=0xfff, blendFunction=three.BlendFunction.ALPHA),
            #                 ]).prop(autoClear=False),
            #     three.Mesh([
            #         three.BoxGeometry(),
            #         three.Edges(),
            #         three.MeshStandardMaterial().prop(color="orange", opacity=0.5, transparent=True),
            #     ]).prop(enableSelect=True, castShadow=True, position=(0, 0, 0), enableHover=True, 
            #         enablePivotControl=True,
            #         enablePivotControlOnSelected=True,
            #         pivotControlProps=three.PivotControlsCommonProps(depthTest=False, annotations=True, anchor=(0, 0, 0))
            #         ),

            # ]),
            three.PivotControls([
                three.Mesh([
                    three.BoxGeometry(),
                    three.Edges(),
                    three.MeshStandardMaterial().prop(color="orange", opacity=0.5, transparent=True),
                ]).prop(enableSelect=True, castShadow=True, position=(5, 0, 0)),            
            ]).prop(anchor=(1, 1, 1), depthTest=False, annotations=True, fixed=True, scale=60),
            

            three.Button("Click Me!", 8, 3, lambda : print("Clicked!")).prop(position=(0, 5, 1)),
        ])
        canvas.canvas.prop(shadows=True)
        return mui.VBox([
            canvas.prop(flex=1),
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden")


async def _main():
    url = "https://uploads.codesandbox.io/uploads/user/b3e56831-8b98-4fee-b941-0e27f39883ab/or72-porsche-transformed.glb"

    data = await download_file_Dev(url)


if __name__ == "__main__":
    asyncio.run(_main())