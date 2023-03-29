import json
import sys 
from tensorpc.serve.__main__ import serve_in_terminal
import base64
from pathlib import Path 

def main():
    enc_b64 = sys.argv[1]
    serv_option_str = base64.b64decode(enc_b64).decode("utf-8")
    serv_option = json.loads(serv_option_str)
    assert "module" in serv_option, "you must provide single module and other serve args"
    may_be_path = sys.argv[2]
    assert Path(may_be_path).exists(), "you must provide a valid path"
    module = serv_option["module"]
    if "serv_config_b64" in serv_option:
        serv_config_b64 = serv_option["serv_config_b64"]
        serv_config = json.loads(base64.b64decode(serv_config_b64).decode("utf-8"))
        serv_config[module]["external_argv"] = sys.argv[2:]
        serv_config_b64 = base64.b64encode(json.dumps(serv_config).encode("utf-8")).decode("utf-8")
        serv_option["serv_config_b64"] = serv_config_b64
        serv_option.pop("module")
    serve_in_terminal(module, **serv_option)

if __name__ == "__main__":
    main()