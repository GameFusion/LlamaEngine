import os
import subprocess
import shutil

import sys

REL_PATH = "build/Qt_6_7_0_for_macOS-Release/EchoLlama.app/Contents/Resources/"
debug=False

if len(sys.argv) > 1:
    if sys.argv[1] == 'debug':
        print("Debug mode is enabled")
        REL_PATH = "build/Qt_6_7_0_for_macOS-Debug/EchoLlama.app/Contents/Resources/"
        debug = True
    elif sys.argv[1] == 'release':
        print("Release mode enabled")
        REL_PATH = "build/Qt_6_7_0_for_macOS-Release/EchoLlama.app/Contents/Resources/"
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
BUILD_ROOT = "../../LlamaEngine/build/Qt_6_7_0_for_macOS-Release/"
if debug:
    BUILD_ROOT = "../../LlamaEngine/build/Qt_6_7_0_for_macOS-Debug/"
OUTPUT_ROOT = REL_PATH + "llama.cpp"

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

def updateLibReferences(old_path, new_path, export_dylib):
    #old_path = f"@rpath/{lib}"
    #new_path = f"@loader_path/{lib}"
    cmd = ["install_name_tool", "-change", old_path, new_path, export_dylib]
    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] Updated {export_dylib}: {old_path} → {new_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error updating {export_dylib}: {e}")
        exit(0)

def updateLibId(new_id, export_dylib):
    cmd = ["install_name_tool", "-id", new_id, export_dylib]
    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] Updated {export_dylib}: id → {new_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error updating {export_dylib}: {e}")
        exit(0)
        #install_name_tool -id @loader_path/LlamaEngine.dylib /path/to/your/LlamaEngine.dylib

def update_rpath_to_loaderpath(dylib_path):
    # Step 1: List all dependencies of the dylib with otool
    try:
        result = subprocess.run(
            ['otool', '-L', dylib_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error listing dependencies: {result.stderr}")
            return

        # Step 2: Process each dependency line to find and update rpath
        lines = result.stdout.splitlines()
        for line in lines:
            # Check if the line contains 'rpath'
            if 'rpath' in line:
                # Extract the path part
                dep_path = line.split()[0]
                
                # Generate the new path with @loader_path
                new_dep_path = f"@loader_path/{os.path.basename(dep_path)}"
                
                # Update the dylib to use @loader_path instead of rpath
                print(f"Updating: {dep_path} -> {new_dep_path}")
                subprocess.run(
                    ['install_name_tool', '-change', dep_path, new_dep_path, dylib_path],
                    check=True
                )

        print("All rpath references updated to @loader_path.")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# Create directory structure and copy DLLs
def organize_builds(commit_version):
    versioned_dir = os.path.join(OUTPUT_ROOT, commit_version)
    os.makedirs(versioned_dir, exist_ok=True)

    build_types = ["metal"]
    for build in build_types:
        src_bin = os.path.join(BUILD_ROOT, build, "bin")
        #dest_dir = os.path.join(versioned_dir, build)
        dest_dir = os.path.join(versioned_dir, build)
        print("dest_dir", dest_dir)
        
        #if os.path.exists(src_bin):
        #    os.makedirs(dest_dir, exist_ok=True)
        #    for file in os.listdir(src_bin):
        #        if file.endswith(".dll"):
        #            shutil.copy(os.path.join(src_bin, file), dest_dir)
        #
        #    print(f"[+] Copied {build} DLLs to {dest_dir}")
        #else:
        #    print("skipping", src_bin)

        print("BUILD_ROOT", BUILD_ROOT)
        
        import_dylib = os.path.join(BUILD_ROOT, "bin", build, "libLlamaEngine.1.dylib")
        export_dylib = os.path.join(dest_dir, "libLlamaEngine.1.dylib")
        
        print("import_dll", import_dylib)
        print("export_dylib", export_dylib)

        # install_name_tool -change libLlamaEngine.1.dylib @loader_path/LlamaEngine.dylib /path/to/your/LlamaEngine.dylib
        #updateLibReferences(f"libLlamaEngine.1.dylib", f"@loader_path/LlamaEngine.dylib", export_dylib)
        #updateLibId(f"@loader_path/LlamaEngine.dylib", export_dylib)
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(export_dylib), exist_ok=True)

        # Copy the file with the new name specified in export_dylib
        shutil.copy(import_dylib, export_dylib)
        print(f"[+] Copied {import_dylib} to {export_dylib}")

        #new_id = "@loader_path/LlamaEngine.dylib"  # This is a common example of using loader_path
        #export_dylib = "/path/to/your/LlamaEngine.dylib"  # Path to the actual dylib you want to modify
        #updateLibId(new_id, export_dylib)

        llama_cpp_libs = ["libllama.dylib", "libggml.dylib", "libggml-base.dylib", "libggml-blas.dylib", "libggml-cpu.dylib", "libggml-metal.dylib"]

        for lib in llama_cpp_libs:
            src_dylib = os.path.join(LLAMA_REPO, "build", "bin", lib)
            dst_dylib = os.path.join(dest_dir, lib)
            shutil.copy(src_dylib, dst_dylib)
            print(f"[+] Copied {src_dylib} to {dst_dylib}")
            
            updateLibReferences(f"@rpath/{lib}", f"@loader_path/{lib}", export_dylib)
            update_rpath_to_loaderpath(dst_dylib)
            
        print("[✓] Updated dylib dependencies")
        subprocess.run(["otool", "-L", export_dylib])
# Main execution
if __name__ == "__main__":
    commit_version, commit_date, commit_hash = get_git_info()
    generate_header(commit_version, commit_date, commit_hash)
    organize_builds(commit_version)
    print("[✓] Build organization complete!")
