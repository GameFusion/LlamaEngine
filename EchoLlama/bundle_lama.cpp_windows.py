import os
import subprocess
import shutil

import sys

REL_PATH = "../build-EchoLlama-Desktop_Qt_5_15_0_MSVC2019_64bit-Release/release/"
debug=False

if len(sys.argv) > 1:
    if sys.argv[1] == 'debug':
        print("Debug mode is enabled")
        REL_PATH = "../build-EchoLlama-Desktop_Qt_5_15_0_MSVC2019_64bit-Debug/debug/"
        debug = True
    elif sys.argv[1] == 'release':
        print("Release mode enabled")
        REL_PATH = "../build-EchoLlama-Desktop_Qt_5_15_0_MSVC2019_64bit-Release/release/"
    elif sys.argv[1] == 'test':
        print("Test mode enabled")
    elif sys.argv[1] == 'debugtest':
        print("Test mode enabled")
        debug = True
    else:
        print("Undefined mode")
        exit(0)


# Paths
LLAMA_REPO = "../../../ExternalCode/llama.cpp"
BUILD_ROOT = "../../../Programmes/llama.cpp"
OUTPUT_ROOT = REL_PATH + "llama.cpp"

if debug:
    LLAMA_ENGINE_ROOT = "../../build-LlamaEngine-Desktop_Qt_6_6_1_MSVC2019_64bit-Debug"
else:
    LLAMA_ENGINE_ROOT = "../../build-LlamaEngine-Desktop_Qt_6_6_1_MSVC2019_64bit-Release"

# Ensure the output directory exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Get llama.cpp version info
def get_git_info():
    print("get_git_info")
    commit_version = subprocess.check_output(["git", "describe"], cwd=LLAMA_REPO, text=True).strip()
    print("commit_version", commit_version)
    commit_date = subprocess.check_output(["git", "log", "-1", "--format=%cd"], cwd=LLAMA_REPO, text=True).strip()
    print("commit_date", commit_date)
    commit_hash = subprocess.check_output(["git", "log", "-1", "--format=%H"], cwd=LLAMA_REPO, text=True).strip()
    print("commit_hash", commit_hash)
    return commit_version, commit_date, commit_hash

# Generate llama_version.h
def generate_header(commit_version, commit_date, commit_hash):
    header_content = f"""#ifndef LLAMA_VERSION_H
#define LLAMA_VERSION_H

#define LLAMA_COMMIT_VERSION "{commit_version}"
#define LLAMA_COMMIT_DATE "{commit_date}"
#define LLAMA_COMMIT_HASH "{commit_hash}"

#endif // LLAMA_VERSION_H
"""
    with open(os.path.join(OUTPUT_ROOT, "llama_version.h"), "w") as f:
        f.write(header_content)
    with open(os.path.join("../", "llama_version.h"), "w") as f:
        f.write(header_content)
    print("[+] Created llama_version.h")

# Create directory structure and copy DLLs
def organize_builds(commit_version):
    versioned_dir = os.path.join(OUTPUT_ROOT, commit_version)
    os.makedirs(versioned_dir, exist_ok=True)

    #build_types = ["cpu", "cuda"]
    build_types = ["cpu", "cuda", "vulkan"]
    for build in build_types:
        src_bin = os.path.join(BUILD_ROOT, build, "bin")
        dest_dir = os.path.join(versioned_dir, build)
        if os.path.exists(src_bin):
            os.makedirs(dest_dir, exist_ok=True)
            for file in os.listdir(src_bin):
                if file.endswith(".dll"):
                    shutil.copy(os.path.join(src_bin, file), dest_dir)

            print(f"[+] Copied {build} DLLs to {dest_dir}")
        else:
            print("skipping", src_bin)

        if debug:
            import_dll = os.path.join(LLAMA_ENGINE_ROOT, "bin", build, "LlamaEngined.dll")
        else:
            import_dll = os.path.join(LLAMA_ENGINE_ROOT, "bin", build, "LlamaEngine.dll")

        shutil.copy(import_dll, dest_dir)
        print(f"[+] Copied {import_dll} DLLs to {dest_dir}")

# Main execution
if __name__ == "__main__":
    commit_version, commit_date, commit_hash = get_git_info()
    generate_header(commit_version, commit_date, commit_hash)
    organize_builds(commit_version)
    print("[✓] Build organization complete!")
