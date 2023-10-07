import enum 

class TextureFormat(enum.IntEnum):
    AlphaFormat = 0
    RedFormat = 1
    RedIntegerFormat = 2
    RGFormat = 3
    RGIntegerFormat = 4
    RGBFormat = 5
    RGBIntegerFormat = 6
    RGBAFormat = 7
    RGBAIntegerFormat = 8
    LuminanceFormat = 9
    LuminanceAlphaFormat = 10
    DepthFormat = 11
    DepthStencilFormat = 12

class TextureType(enum.IntEnum):
    UnsignedByteType = 0
    ByteType = 1
    ShortType = 2
    UnsignedShortType = 3
    IntType = 4
    UnsignedIntType = 5
    FloatType = 6
    HalfFloatType = 7
    UnsignedShort4444Type = 8
    UnsignedShort5551Type = 9
    UnsignedShort565Type = 10
    UnsignedInt248Type = 11

class TextureMappingType(enum.Enum):
    UVMapping = 0
    CubeReflectionMapping = 1
    CubeRefractionMapping = 2
    EquirectangularReflectionMapping = 3
    EquirectangularRefractionMapping = 4
    CubeUVReflectionMapping = 5

class TextureWrappingMode(enum.IntEnum):
    RepeatWrapping = 0
    ClampToEdgeWrapping = 1
    MirroredRepeatWrapping = 2
    
class TextureFilterType(enum.IntEnum):
    NearestFilter = 0
    NearestMipmapNearestFilter = 1
    NearestMipmapLinearFilter = 2
    LinearFilter = 3
    LinearMipmapNearestFilter = 4
    LinearMipmapLinearFilter = 5
