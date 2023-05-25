import re
import subprocess
from pathlib import Path
# from tensorpc.constants import PACKAGE_ROOT
# import codeai


def compile_proto(cwd,
                  proto_dir,
                  js_out=True,
                  cpp_out=False,
                  grpc_web: bool = False,
                  with_pyi: bool = True):
    proto_dir_p = Path(proto_dir)
    proto_files = list(Path(proto_dir).glob("*.proto"))
    grpc_files = ["remote_object.proto"]
    grpc_paths = [proto_dir_p / p for p in grpc_files]
    no_grpc_path = list(filter(lambda x: x.name not in grpc_files,
                               proto_files))
    grpc_proto_cmds = [
        "python",
        "-m",
        "grpc_tools.protoc",
        "-I{}".format(proto_dir),
        "--python_out={}".format(proto_dir),
        "--pyi_out={}".format(proto_dir) if with_pyi else "",
        "--grpc_python_out={}".format(proto_dir),
        *[str(p) for p in grpc_paths],  # windows have problem with wildcard
    ]
    no_grpc_proto_cmds = [
        "python",
        "-m",
        "grpc_tools.protoc",
        "-I{}".format(proto_dir),
        "--python_out={}".format(proto_dir),
        "--pyi_out={}".format(proto_dir) if with_pyi else "",
        *[str(p) for p in no_grpc_path],  # windows have problem with wildcard
    ]
    cpp_proto_dir = str(Path(proto_dir) / "cpp")
    js_proto_dir = str(Path(proto_dir) / "js")
    cmds_js = [
        "protoc", f"-I={proto_dir}", f"{proto_dir}/*.proto",
        "--js_out=import_style=commonjs:{} ".format(js_proto_dir)
    ]
    # grpc_web_cmds = []
    if grpc_web:
        cmds_js.append(
            "--grpc-web_out=import_style=commonjs+dts,mode=grpcwebtext:{}".
            format(js_proto_dir))
        # grpc_web_cmds = ["--plugin=protoc-gen-grpc=\"{}\"".format(protoc_cpp_path),]
    if cpp_out:
        protoc_cpp_path = subprocess.check_output(["which", "grpc_cpp_plugin"])
        protoc_cpp_path = protoc_cpp_path.decode("utf-8").strip()
        cpp_cmds = [
            "protoc",
            "-I{}".format(proto_dir),
            "--cpp_out={}".format(cpp_proto_dir),
            "--grpc_out={}".format(cpp_proto_dir),
            "--plugin=protoc-gen-grpc=\"{}\"".format(protoc_cpp_path),
            *[str(p) for p in proto_files],
        ]
        output = subprocess.check_output(" ".join(cpp_cmds),
                                         shell=True,
                                         cwd=str(cwd))
    output = subprocess.check_output(" ".join(grpc_proto_cmds),
                                     shell=True,
                                     cwd=str(cwd))
    output = subprocess.check_output(" ".join(no_grpc_proto_cmds),
                                     shell=True,
                                     cwd=str(cwd))

    if js_out:
        output = subprocess.check_output(" ".join(cmds_js),
                                         shell=True,
                                         cwd=str(cwd))

    print(output)
    # TODO: add eslint-disable to js outputs?
    proto_dir = Path(proto_dir)
    pb_file_pattern = re.compile(".*_pb2")
    grpc_pb_file_pattern = re.compile(".*_pb2_grpc")
    proto_pkg_names = []
    for path in proto_dir.glob("*.py"):
        if pb_file_pattern.fullmatch(path.stem):
            proto_pkg_names.append(path.stem[:-4])
    pb_proto_pkg_names = [s + "_pb2_grpc" for s in proto_pkg_names]
    grpc_proto_pkg_names = [s + "_pb2" for s in proto_pkg_names]
    import_as_pattern = re.compile(r"import (?:{}) as .*\n".format(
        "|".join(pb_proto_pkg_names + grpc_proto_pkg_names)))
    print(r"import (?:{}) as .*".format("|".join(pb_proto_pkg_names +
                                                 grpc_proto_pkg_names)))
    for path in proto_dir.glob("*.py"):
        if pb_file_pattern.fullmatch(
                path.stem) or grpc_pb_file_pattern.fullmatch(path.stem):
            with path.open("r") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                if import_as_pattern.fullmatch(lines[i]):
                    print(lines[i])
                    lines[i] = "from . " + lines[i]
            with path.open("w") as f:
                f.writelines(lines)

def main_legacy():
    compile_proto(
        Path(__file__).parent / "protos_legacy",
        Path(__file__).parent.resolve() / "protos_legacy",
        js_out=False,
        with_pyi=False)

def main():
    compile_proto(
        Path(__file__).parent / "protos",
        Path(__file__).parent.resolve() / "protos",
        js_out=False)


if __name__ == "__main__":
    main()
