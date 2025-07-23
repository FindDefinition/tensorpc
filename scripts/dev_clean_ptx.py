import fire 


def main(path: str, out_path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    # only keep ld.*, st.*, ldmatrix.*, stmatrix.*, mma.sync.aligned.* and cp.async
    keep_keywords = [
        "ld.global", "ld.shared", "st.global", "st.shared",
        "ldmatrix", "stmatrix", "mma.sync.aligned", "cp.async",
        "bar.sync",
    ]
    filtered_lines = []
    # if line contains any of the keep_keywords, keep it
    for line in lines:
        if any(keyword in line for keyword in keep_keywords):
            filtered_lines.append(line)
    # write to out path
    with open(out_path, 'w') as f:
        f.writelines(filtered_lines)
    print(f"Filtered PTX code written to {out_path}")

if __name__ == "__main__":
    fire.Fire(main)