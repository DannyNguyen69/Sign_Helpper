import os
import subprocess
import sys
import mediapipe


def build_exe():
    script_name = "App_combined.py"
    model_file = "MLP_model.p"
    if not os.path.exists(script_name):
        print(f"Không tìm thấy file {script_name}")
        return
    if not os.path.exists(model_file):
        print(f"Không tìm thấy file {model_file}")
        return

    # Xóa build/dist/spec cũ nếu có
    for folder in ["build", "dist"]:
        if os.path.exists(folder):
            import shutil
            shutil.rmtree(folder)
    spec_file = "HandSignApp.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)

    # Thêm các hidden-import phổ biến cho sklearn, scipy, joblib, numpy và _forest
    hidden_imports = [
        "sklearn", "sklearn.utils._weight_vector", "sklearn.utils._openmp_helpers",
        "sklearn.ensemble._forest",  
        "scipy", "joblib", "numpy", "numpy.core._methods", "numpy.lib.format"
    ]
    modules_path = os.path.join(os.path.dirname(mediapipe.__file__), "modules")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--add-data", f"{model_file};.",
        "--add-data", f"{modules_path};mediapipe/modules",  
        "--name", "HandSignApp",
    ]
    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])
    cmd.append(script_name)

    print("Đang build file exe, vui lòng chờ.")
    subprocess.run(cmd, check=True)
    print(" Đã build xong File exe nằm trong thư mục 'dist'.")

if __name__ == "__main__":
    install_pyinstaller()
    build_exe() 