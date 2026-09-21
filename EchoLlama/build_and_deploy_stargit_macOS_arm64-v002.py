import os
import sys
import subprocess
import shutil
import datetime
import hashlib

# Add the parent directory to the Python path for importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import S3  # Import the S3 module for uploading
#import SQL  # Import the SQL module for database operations
#import TeamsMessaging  # Import the Microsoft Teams notifications

# Generate a unique name using the current date and time
x = datetime.datetime.now()
snapshot_datetime = x.strftime("%Y-%m-%d_%H-%M")
snapshot_datetime_pretty = x.strftime("%Y-%m-%d %H:%M")

# Set the base name for the app and DMG
app_base_name = f"EchoLlama-{snapshot_datetime}-macOS-arm64"
dmg_name = f"{app_base_name}.dmg"

# Paths for build output and DMG creation
build_folder = f"build/Qt_6_7_0_for_macOS-Release/EchoLlama.app"
destination_folder = app_base_name

def run_command(cmd, continue_on_error=False):
    """Utility function to run shell commands with real-time output."""
    print(f"Running: {' '.join(cmd)}")

    # Use Popen to handle real-time streaming of stdout and stderr
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read stdout and stderr as they are generated
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line, end='')  # Print stdout in real-time

    for stderr_line in iter(process.stderr.readline, ""):
        print(stderr_line, end='', file=sys.stderr)  # Print stderr in real-time

    process.stdout.close()
    process.stderr.close()

    return_code = process.wait()  # Wait for the process to finish

    # Handle non-zero exit codes
    if return_code != 0:
        print(f"Error: Command returned non-zero exit code {return_code}")
        if not continue_on_error:
            sys.exit(return_code)  # Exit immediately on error
        else:
            print("Continuing despite the error...")

def md5sum(filename):
    """Calculate MD5 checksum of a file"""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def insert_into_db(myDict):
    """Insert the metadata into the SQL database"""
    db = SQL.connect(user="root")
    SQL.insert(db, myDict)  # Use the `insert` function from the SQL module

def sign_directory_recursive(directory_path, sign_identity):
    """
    Recursively signs all files in the given directory and its subdirectories.
    Skips directories themselves but processes their contents.
    """
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):  # Only sign files
            run_command([
                "codesign", "--force", "--verify", "--verbose", "--sign",
                sign_identity, "--options", "runtime", item_path
            ])
            print("[✓] codesign complete:", item_path)
        elif os.path.isdir(item_path):  # Recurse into subdirectories
            sign_directory_recursive(item_path, sign_identity)

def build_macOS_application():
    """Build and package the macOS StarGit application"""
    sign_identity = "Developer ID Application: GameFusion SAS (W9Y52TA3WD)"

    # Run macdeployqt
    print("macdeployqt")
    run_command([os.path.expanduser("~/Qt/6.7.0/macos/bin/macdeployqt"), build_folder])

    # Remove extended attributes
    print("xattr")
    run_command(["xattr", "-cr", build_folder])

    # Remove temporary .cstemp files
    print("Remove temporary files")
    frameworks_path = f"{build_folder}/Contents/Frameworks"
    for root, dirs, files in os.walk(frameworks_path):
        for file in files:
            if file.endswith(".cstemp"):
                file_path = os.path.join(root, file)
                print(f"Removing temporary file: {file_path}")
                os.remove(file_path)

    # Sign frameworks
    frameworks_path = f"{build_folder}/Contents/Frameworks"
    print("frameworks_path:", frameworks_path)
    for lib in os.listdir(frameworks_path):
        lib_path = os.path.join(frameworks_path, lib)
        print("-> signing:", lib_path)
        run_command(["codesign", "--force", "--verify", "--verbose", "--sign", sign_identity, "--options", "runtime", lib_path])
        print("[✓] codesign complete:", lib_path)

    # Sign plugins folder
    plugins_path = f"{build_folder}/Contents/PlugIns"
    print("sign_directory_recursive")
    sign_directory_recursive(plugins_path, sign_identity)


    #exit(0)

    # Sign the app binary and individual Qt frameworks
    binaries_to_sign = [
        f"{frameworks_path}/QtNetwork.framework/QtNetwork",
        f"{frameworks_path}/QtWidgets.framework/QtWidgets",
        f"{frameworks_path}/QtCore.framework/QtCore",
        f"{frameworks_path}/QtDBus.framework/QtDBus",
        f"{frameworks_path}/QtGui.framework/QtGui"
    ]

    #binaries_to_sign.extend(dylib_files)
    print("sign binaries in frameworks")
    for binary in binaries_to_sign:
        run_command(["codesign", "--force", "--verify", "--verbose", "--sign", sign_identity, "--options", "runtime", binary])
        print("[✓] codesign complete:", binary)
    #exit(0)

    #
    #
    # Sign plugins
    plugins_path = f"{build_folder}/Contents/PlugIns"
    print("sign binaries in plugins")
    for root, dirs, files in os.walk(plugins_path):
        for plugin in files:
            plugin_path = os.path.join(root, plugin)
            run_command(["codesign", "--force", "--verify", "--verbose", "--sign", sign_identity, "--options", "runtime", plugin_path])
            print("[✓] codesign complete:", binary)


    #
    #
    # Sign the dylib files

     # List all .dylib files under Resources/llama.cpp, recursively using os.walk()
    dylib_files = []
     # Walk through directories in build_folder
    print("build_folder:", build_folder)
    print("sign binaries in llama.cpp")
    for root, dirs, files in os.walk(f"{build_folder}/Contents/Resources/llama.cpp"):
        print("root:", root)
        print("dirs:", dirs)
        print("files:", files)
        for file in files:
            if file.endswith(".dylib"):
                dylib_files.append(os.path.join(root, file))


    # Print the found dylib files
    print("dylib_files:", dylib_files)

    # Sign the dylib files
    for dylib in dylib_files:
        run_command(["codesign", "--force", "--verify", "--verbose", "--sign", sign_identity, "--options", "runtime", dylib])
        print("[✓] codesign complete:", dylib)


    #
    #
    # Sign the main app binary
    main_app_bin = f"{build_folder}/Contents/MacOS/EchoLlama"
    run_command(["codesign", "--force", "--verify", "--verbose", "--sign", sign_identity, "--options", "runtime", main_app_bin])
    print("[✓] codesign complete:", binary)

    # Build DMG
    #shutil.copytree(build_folder, f"{app_base_name}/StarGit.app")
    run_command(["mkdir", f"{app_base_name}"])
    run_command(["cp", "-Rfpv", build_folder, f"{app_base_name}"])

    #exit(0)


    run_command(["hdiutil", "create", "-volname", "StarGit", "-srcfolder", app_base_name, "-ov", "-format", "UDZO", dmg_name])

    # Sign the DMG
    run_command(["codesign", "--sign", sign_identity, dmg_name])
    print("[✓] codesign complete:", dmg_name)

    #exit(0)

    # Notarize the DMG
    run_command([
        "xcrun", "notarytool", "submit", dmg_name,
        "--key", "../../GitExplorer/build-vs2019-qt6/AuthKey_A4VWY4L3AA.p8",
        "--key-id", "A4VWY4L3AA",
        "--issuer", "69a6de6f-141f-47e3-e053-5b8c7c11a4d1",
        "--team-id", "W9Y52TA3WD", "--wait"
    ])



    # Staple notarization
    run_command(["xcrun", "stapler", "staple", dmg_name])

    # Verify DMG
    #run_command(["spctl", "-a", "-vvv", dmg_name])
    run_command(["spctl", "-a", "-vvv", dmg_name], continue_on_error=True)

    # Upload to S3 using the `S3.py` module
    if False:
        path_s3 = f"Applications/EchoLlama/{dmg_name}"
        url = S3.upload_file(dmg_name, path_s3)

        # Get file metadata for SQL insertion
        snapshot_datetime_pretty = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_size = str(os.path.getsize(dmg_name))
        file_md5 = md5sum(dmg_name)

        # Metadata dictionary for SQL insertion
        myDict = {
            'User': 'Andreas J. Carlen',
            'Name': 'EchoLlama',
            'File': dmg_name,
            'Status': 'pdr',
            'Task': 'beta',
            'Date': snapshot_datetime_pretty,
            'URL': url,
            'Size': file_size,
            'MD5': file_md5,
            'Platform': 'AppleSilicon'
        }

        # Insert into SQL
        insert_into_db(myDict)

        # Post a message to Microsoft Teams (if enabled)
        enable_teams = True
        if enable_teams:
            title = "Application StarGit published to cloud"
            text = f"Application StarGit published on {snapshot_datetime_pretty}"
            TeamsMessaging.post(title, text, 'App Link', url)

if __name__ == "__main__":
    build_macOS_application()