from tensorpc.dock.components import three


def get_mask2d_shader_material(
        distance: float,
        color1: str = "silver",
        color2: str = "white",
        opacity1: float = 0.6,
        opacity2: float = 0.3,
        coeff: float = 1.0) -> three.MeshShaderMaterial:
    return three.MeshShaderMaterial([
        three.ShaderUniform("distance", three.ShaderUniformType.Number, distance),
        three.ShaderUniform("color1", three.ShaderUniformType.Color, color1),
        three.ShaderUniform("color2", three.ShaderUniformType.Color, color2),
        three.ShaderUniform("opacity1", three.ShaderUniformType.Number, opacity1),
        three.ShaderUniform("opacity2", three.ShaderUniformType.Number, opacity2),
        three.ShaderUniform("coeff", three.ShaderUniformType.Number, coeff),
    ], f"""
    varying vec3 worldPosition;

    void main() {{
        worldPosition = (instanceMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    }}
    """, f"""
    varying vec3 worldPosition;
    uniform vec3 color1;
    uniform vec3 color2;
    uniform float opacity1;
    uniform float opacity2;
    uniform float distance;
    uniform float coeff;

    void main() {{
        float c = worldPosition.y - coeff * worldPosition.x;
        // unify c to [0, distance] range (like c % distance)
        c = mod(c + distance, distance);
        vec4 color1Alpha = vec4(color1, opacity1);
        vec4 color2Alpha = vec4(color2, opacity2);
        vec4 color = c > distance / 2.0 ? color1Alpha : color2Alpha;
        gl_FragColor = color;
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
    }}
    """)
