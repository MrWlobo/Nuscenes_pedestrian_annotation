import json
import os
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from PIL import Image, ImageTk
from pyquaternion import Quaternion


class NuScenesAnnotator:
    def __init__(self, master, dataroot, version, out_dir):
        self.master = master
        self.master.title("NuScenes 3D Box Annotator")
        self.dataroot = dataroot
        self.version = version
        self.out_dir = out_dir

        self.out_version_dir = os.path.join(self.out_dir, self.version)
        self.in_version_dir = os.path.join(self.dataroot, self.version)
        self.progress_file = os.path.join(self.out_dir, "progress.json")

        self.setup_workspace()

        print("Loading NuScenes dataset...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = self.nusc.sample

        raw_ann_path = os.path.join(self.out_version_dir, "sample_annotation.json")
        with open(raw_ann_path, "r") as f:
            self.raw_annotations = json.load(f)

        self.raw_ann_map = {ann["token"]: ann for ann in self.raw_annotations}

        self.corrected_tokens = self.load_progress()
        if self.corrected_tokens:
            print(
                f"Found {len(self.corrected_tokens)} corrected samples. Syncing previous edits to memory..."
            )
            for mem_ann in self.nusc.sample_annotation:
                raw_ann = self.raw_ann_map.get(mem_ann["token"])
                if raw_ann:
                    mem_ann["translation"] = raw_ann["translation"]
                    mem_ann["size"] = raw_ann["size"]
                    mem_ann["rotation"] = raw_ann["rotation"]

        self.current_anns = []
        self.selected_ann_idx = None
        self.cam_labels = {}

        self.cameras = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        self.setup_ui()

        self.load_next_uncorrected(start_idx=0)

    def setup_workspace(self):
        """Ensures output directory has all the necessary JSON files."""
        os.makedirs(self.out_version_dir, exist_ok=True)

        if os.path.exists(self.in_version_dir):
            for file in os.listdir(self.in_version_dir):
                if file.endswith(".json"):
                    src = os.path.join(self.in_version_dir, file)
                    dst = os.path.join(self.out_version_dir, file)

                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

    def load_progress(self):
        """Loads the set of already corrected sample tokens."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                return set(json.load(f))
        return set()

    def save_progress(self):
        """Saves the set of corrected sample tokens."""
        with open(self.progress_file, "w") as f:
            json.dump(list(self.corrected_tokens), f)

    def setup_ui(self):

        self.cam_frame = tk.Frame(self.master)
        self.cam_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for i, cam in enumerate(self.cameras):
            row, col = divmod(i, 3)
            lbl = tk.Label(self.cam_frame, text=cam, compound=tk.TOP)
            lbl.grid(row=row, column=col, padx=5, pady=5)
            self.cam_labels[cam] = lbl

        self.ctrl_frame = tk.Frame(self.master)
        self.ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.progress_frame = tk.Frame(self.ctrl_frame)
        self.progress_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, variable=self.progress_var, maximum=len(self.samples)
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_label = tk.Label(self.progress_frame, text="")
        self.progress_label.pack(side=tk.RIGHT, padx=10)
        self.update_progress_ui()

        list_frame = tk.Frame(self.ctrl_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(list_frame, text="Annotations").pack()
        self.ann_listbox = tk.Listbox(list_frame, width=40, height=10)
        self.ann_listbox.pack(side=tk.LEFT, fill=tk.Y)
        self.ann_listbox.bind("<<ListboxSelect>>", self.on_ann_select)

        self.input_frame = tk.Frame(self.ctrl_frame)
        self.input_frame.pack(side=tk.LEFT, padx=20)

        self.vars = {
            "x": tk.DoubleVar(),
            "y": tk.DoubleVar(),
            "z": tk.DoubleVar(),
            "w": tk.DoubleVar(),
            "l": tk.DoubleVar(),
            "h": tk.DoubleVar(),
            "yaw": tk.DoubleVar(),
        }

        for var in self.vars.values():
            var.trace_add("write", self.on_input_change)

        labels = [
            "X (m)",
            "Y (m)",
            "Z (m)",
            "Width (m)",
            "Length (m)",
            "Height (m)",
            "Yaw (deg)",
        ]
        keys = ["x", "y", "z", "w", "l", "h", "yaw"]

        for i, (lbl, key) in enumerate(zip(labels, keys)):
            row, col = divmod(i, 2)
            tk.Label(self.input_frame, text=lbl).grid(
                row=row, column=col * 2, sticky=tk.E, padx=5, pady=2
            )
            tk.Entry(self.input_frame, textvariable=self.vars[key], width=10).grid(
                row=row, column=col * 2 + 1, padx=5, pady=2
            )

        btn_frame = tk.Frame(self.ctrl_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Button(
            btn_frame,
            text="< Prev (Any)",
            command=lambda: self.load_sample(self.current_sample_idx - 1),
        ).pack(fill=tk.X, pady=2)
        tk.Button(
            btn_frame,
            text="Skip to Next >",
            command=lambda: self.load_next_uncorrected(self.current_sample_idx + 1),
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            btn_frame,
            text="Save & Next",
            command=self.save_and_next,
            bg="green",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(fill=tk.X, pady=10)

    def update_progress_ui(self):
        completed = len(self.corrected_tokens)
        total = len(self.samples)
        self.progress_var.set(completed)
        self.progress_label.config(text=f"{completed} / {total} Corrected")

    def load_next_uncorrected(self, start_idx):
        for idx in range(start_idx, len(self.samples)):
            token = self.samples[idx]["token"]
            if token not in self.corrected_tokens:
                self.load_sample(idx)
                return

        for idx in range(0, start_idx):
            token = self.samples[idx]["token"]
            if token not in self.corrected_tokens:
                self.load_sample(idx)
                return

        messagebox.showinfo("Done!", "All samples have been marked as corrected!")

    def load_sample(self, idx):
        if idx < 0 or idx >= len(self.samples):
            return

        self.current_sample_idx = idx
        sample = self.samples[idx]

        self.current_anns = [
            self.nusc.get("sample_annotation", token) for token in sample["anns"]
        ]
        self.ann_listbox.delete(0, tk.END)
        for ann in self.current_anns:
            self.ann_listbox.insert(
                tk.END, f"{ann['category_name']} [{ann['token'][:6]}]"
            )

        self.selected_ann_idx = None

        status = "[CORRECTED] " if sample["token"] in self.corrected_tokens else ""
        self.master.title(
            f"{status}NuScenes Annotator - Sample {idx + 1}/{len(self.samples)}"
        )

        self.render_cameras()

    def on_ann_select(self, event):
        selection = self.ann_listbox.curselection()
        if not selection:
            return

        self.selected_ann_idx = selection[0]
        ann = self.current_anns[self.selected_ann_idx]

        q = Quaternion(ann["rotation"])
        yaw = np.degrees(q.yaw_pitch_roll[0])

        for var in self.vars.values():
            var.trace_vdelete("w", var.trace_info()[0][1])

        self.vars["x"].set(round(ann["translation"][0], 3))
        self.vars["y"].set(round(ann["translation"][1], 3))
        self.vars["z"].set(round(ann["translation"][2], 3))
        self.vars["w"].set(round(ann["size"][0], 3))
        self.vars["l"].set(round(ann["size"][1], 3))
        self.vars["h"].set(round(ann["size"][2], 3))
        self.vars["yaw"].set(round(yaw, 2))

        for var in self.vars.values():
            var.trace_add("write", self.on_input_change)

        self.render_cameras()

    def on_input_change(self, *args):
        if self.selected_ann_idx is None:
            return

        try:
            ann = self.current_anns[self.selected_ann_idx]
            ann["translation"] = [
                self.vars["x"].get(),
                self.vars["y"].get(),
                self.vars["z"].get(),
            ]
            ann["size"] = [
                self.vars["w"].get(),
                self.vars["l"].get(),
                self.vars["h"].get(),
            ]

            yaw_rad = np.radians(self.vars["yaw"].get())
            q = Quaternion(axis=[0, 0, 1], angle=yaw_rad)
            ann["rotation"] = q.elements.tolist()

            self.render_cameras()
        except tk.TclError:
            pass

    def render_cameras(self):
        sample = self.samples[self.current_sample_idx]

        for cam in self.cameras:
            cam_token = sample["data"][cam]
            cam_data = self.nusc.get("sample_data", cam_token)
            im_path = os.path.join(self.nusc.dataroot, cam_data["filename"])

            img = cv2.imread(im_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cs_record = self.nusc.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )
            pose_record = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
            camera_intrinsic = np.array(cs_record["camera_intrinsic"])

            for i, ann in enumerate(self.current_anns):
                box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)

                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)

                if box.center[2] < 0.1:
                    continue

                color = (255, 0, 0) if i == self.selected_ann_idx else (0, 255, 0)
                thickness = 3 if i == self.selected_ann_idx else 1

                box.render_cv2(
                    img,
                    view=camera_intrinsic,
                    normalize=True,
                    colors=(color, color, color),
                    linewidth=thickness,
                )

            img_resized = cv2.resize(img, (400, 225))
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.cam_labels[cam].config(image=img_tk)
            self.cam_labels[cam].image = img_tk

    def save_and_next(self):

        current_sample = self.samples[self.current_sample_idx]
        self.corrected_tokens.add(current_sample["token"])

        self.save_progress()
        self.update_progress_ui()

        for ann in self.current_anns:
            raw_ann = self.raw_ann_map[ann["token"]]
            raw_ann["translation"] = ann["translation"]
            raw_ann["size"] = ann["size"]
            raw_ann["rotation"] = ann["rotation"]

        out_path = os.path.join(self.out_version_dir, "sample_annotation.json")
        try:
            with open(out_path, "w") as f:
                json.dump(self.raw_annotations, f, indent=4)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save annotations: {str(e)}")
            return

        self.load_next_uncorrected(self.current_sample_idx + 1)


if __name__ == "__main__":
    root = tk.Tk()

    DATAROOT = "selected_scenes"
    VERSION = "v1.0-trainval"
    OUTPUT_DIR = "corrected_scenes"

    app = NuScenesAnnotator(root, DATAROOT, VERSION, OUTPUT_DIR)
    root.mainloop()
