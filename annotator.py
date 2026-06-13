import json
import os
import shutil
import tkinter as tk
import uuid
from tkinter import messagebox, ttk

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from PIL import Image, ImageTk
from pyquaternion import Quaternion


class CategoryDialog(tk.Toplevel):
    def __init__(self, parent, categories):
        super().__init__(parent)
        self.title("Select Category")
        self.result = None
        self.geometry("300x130")
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="Choose annotation category:").pack(pady=10)
        self.combo = ttk.Combobox(self, values=categories, state="readonly")
        self.combo.pack(pady=5, padx=20, fill=tk.X)
        if categories:
            self.combo.set(categories[0])

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=self.on_ok, width=8).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="Cancel", command=self.destroy, width=8).pack(
            side=tk.LEFT, padx=5
        )
        self.wait_window(self)

    def on_ok(self):
        self.result = self.combo.get()
        self.destroy()


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

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = self.nusc.sample

        raw_ann_path = os.path.join(self.out_version_dir, "sample_annotation.json")
        with open(raw_ann_path, "r") as f:
            self.raw_annotations = json.load(f)
        self.raw_ann_map = {ann["token"]: ann for ann in self.raw_annotations}

        raw_inst_path = os.path.join(self.out_version_dir, "instance.json")
        with open(raw_inst_path, "r") as f:
            self.raw_instances = json.load(f)
        self.raw_inst_map = {inst["token"]: inst for inst in self.raw_instances}

        self.corrected_tokens = self.load_progress()

        self.nusc.sample_annotation = self.raw_annotations
        self.nusc.instance = self.raw_instances

        self.nusc._token2ind["sample_annotation"] = {
            ann["token"]: i for i, ann in enumerate(self.raw_annotations)
        }
        self.nusc._token2ind["instance"] = {
            inst["token"]: i for i, inst in enumerate(self.raw_instances)
        }

        for sample in self.nusc.sample:
            sample["anns"] = []

        for ann in self.raw_annotations:
            sample_token = ann["sample_token"]
            if sample_token in self.nusc._token2ind["sample"]:
                sample_idx = self.nusc._token2ind["sample"][sample_token]
                self.nusc.sample[sample_idx]["anns"].append(ann["token"])

        self.current_anns = []
        self.selected_ann_idx = None
        self.cam_labels = {}
        self._lock_traces = False
        self.keyframe = None

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
        os.makedirs(self.out_version_dir, exist_ok=True)
        if os.path.exists(self.in_version_dir):
            for file in os.listdir(self.in_version_dir):
                if file.endswith(".json"):
                    src = os.path.join(self.in_version_dir, file)
                    dst = os.path.join(self.out_version_dir, file)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                return set(json.load(f))
        return set()

    def save_progress(self):
        with open(self.progress_file, "w") as f:
            json.dump(list(self.corrected_tokens), f)

    def setup_ui(self):
        # Top Frame: Cameras
        self.cam_frame = tk.Frame(self.master)
        self.cam_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for i, cam in enumerate(self.cameras):
            row, col = divmod(i, 3)
            lbl = tk.Label(self.cam_frame, text=cam, compound=tk.TOP)
            lbl.grid(row=row, column=col, padx=2, pady=2)
            self.cam_labels[cam] = lbl

        # Bottom Frame: Master Container
        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Progress Bar Header
        self.progress_frame = tk.Frame(self.bottom_frame)
        self.progress_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, variable=self.progress_var, maximum=len(self.samples)
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_label = tk.Label(self.progress_frame, text="")
        self.progress_label.pack(side=tk.RIGHT, padx=10)

        # Control Panel Grid
        self.ctrl_frame = tk.Frame(self.bottom_frame)
        self.ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        self.ctrl_frame.columnconfigure(0, weight=1)
        self.ctrl_frame.columnconfigure(1, weight=1)
        self.ctrl_frame.columnconfigure(2, weight=1)
        self.ctrl_frame.columnconfigure(3, weight=1)

        # Column 0: Listbox
        list_frame = tk.Frame(self.ctrl_frame)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        tk.Label(list_frame, text="Annotations", font=("Arial", 9, "bold")).pack()
        self.ann_listbox = tk.Listbox(list_frame, width=28, height=8)
        self.ann_listbox.pack(fill=tk.BOTH, expand=True)
        self.ann_listbox.bind("<<ListboxSelect>>", self.on_ann_select)

        # Column 1: Action Buttons (Grid Layout)
        action_frame = tk.Frame(self.ctrl_frame)
        action_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        tk.Label(action_frame, text="Track Actions", font=("Arial", 9, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 2)
        )

        tk.Button(
            action_frame,
            text="Add Box",
            command=self.add_annotation,
            bg="blue",
            fg="white",
            width=12,
        ).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        tk.Button(
            action_frame,
            text="Propagate >",
            command=self.propagate_to_next,
            bg="orange",
            fg="black",
            width=12,
        ).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        tk.Button(
            action_frame,
            text="Delete Track (All)",
            command=self.delete_annotation,
            bg="red",
            fg="white",
        ).grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="ew")

        ttk.Separator(action_frame, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=6
        )

        tk.Label(action_frame, text="Interpolation", font=("Arial", 9, "bold")).grid(
            row=4, column=0, columnspan=2
        )
        self.keyframe_label = tk.Label(
            action_frame, text="Keyframe: None", fg="gray", font=("Arial", 8)
        )
        self.keyframe_label.grid(row=5, column=0, columnspan=2)
        tk.Button(
            action_frame,
            text="Mark Start",
            command=self.mark_start_keyframe,
            bg="purple",
            fg="white",
        ).grid(row=6, column=0, padx=2, pady=2, sticky="ew")
        tk.Button(
            action_frame,
            text="Interp to Here",
            command=self.interpolate_to_here,
            bg="#E91E63",
            fg="white",
        ).grid(row=6, column=1, padx=2, pady=2, sticky="ew")

        # Column 2: Coordinate Inputs
        input_frame = tk.Frame(self.ctrl_frame)
        input_frame.grid(row=0, column=2, sticky="nsew", padx=5)
        tk.Label(input_frame, text="Properties", font=("Arial", 9, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 2)
        )

        self.vars = {k: tk.DoubleVar() for k in ["x", "y", "z", "w", "l", "h", "yaw"]}
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
            tk.Label(input_frame, text=lbl).grid(
                row=row + 1, column=col * 2, sticky=tk.E, padx=2, pady=1
            )
            tk.Entry(input_frame, textvariable=self.vars[key], width=8).grid(
                row=row + 1, column=col * 2 + 1, padx=2, pady=1
            )

        # Column 3: Navigation & Status
        nav_frame = tk.Frame(self.ctrl_frame)
        nav_frame.grid(row=0, column=3, sticky="nsew", padx=5)

        self.status_indicator = tk.Label(
            nav_frame,
            text="UNKNOWN",
            font=("Arial", 11, "bold"),
            pady=4,
            relief=tk.GROOVE,
        )
        self.status_indicator.pack(fill=tk.X, pady=(0, 5))

        tk.Button(
            nav_frame,
            text="Toggle Ready State",
            command=self.toggle_ready_state,
            font=("Arial", 8),
        ).pack(fill=tk.X, pady=(0, 10))

        tk.Button(
            nav_frame,
            text="< Prev Frame",
            command=lambda: self.load_sample(self.current_sample_idx - 1),
        ).pack(fill=tk.X, pady=1)
        tk.Button(
            nav_frame,
            text="Skip to Next >",
            command=lambda: self.load_next_uncorrected(self.current_sample_idx + 1),
        ).pack(fill=tk.X, pady=1)
        tk.Button(
            nav_frame,
            text="Save & Mark Ready",
            command=self.save_and_next,
            bg="green",
            fg="white",
            font=("Arial", 10, "bold"),
            pady=4,
        ).pack(fill=tk.X, pady=(10, 0))

    def update_progress_ui(self):
        completed = len(self.corrected_tokens)
        total = len(self.samples)
        self.progress_var.set(completed)
        self.progress_label.config(text=f"{completed} / {total} Corrected")

    def update_status_indicator(self):
        sample = self.samples[self.current_sample_idx]
        if sample["token"] in self.corrected_tokens:
            self.status_indicator.config(
                text="✅ MARKED AS READY", bg="#c8e6c9", fg="#2e7d32"
            )
        else:
            self.status_indicator.config(
                text="⚠️ NEEDS REVIEW", bg="#ffecb3", fg="#b08d00"
            )

    def toggle_ready_state(self):
        token = self.samples[self.current_sample_idx]["token"]
        if token in self.corrected_tokens:
            self.corrected_tokens.remove(token)
        else:
            self.corrected_tokens.add(token)
        self.save_progress()
        self.update_progress_ui()
        self.update_status_indicator()

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
            cat_name = "unknown"
            try:
                inst = self.nusc.get("instance", ann["instance_token"])
                cat = self.nusc.get("category", inst["category_token"])
                cat_name = cat["name"]
            except Exception:
                pass

            self.ann_listbox.insert(tk.END, f"{cat_name} [{ann['token'][:6]}]")

        self.selected_ann_idx = None
        self.master.title(f"NuScenes Annotator - Sample {idx + 1}/{len(self.samples)}")

        self.update_status_indicator()
        self.render_cameras()

    def on_ann_select(self, event):
        selection = self.ann_listbox.curselection()
        if not selection:
            return

        self.selected_ann_idx = selection[0]
        ann = self.current_anns[self.selected_ann_idx]

        q = Quaternion(ann["rotation"])
        yaw = np.degrees(q.yaw_pitch_roll[0])

        self._lock_traces = True

        self.vars["x"].set(round(ann["translation"][0], 3))
        self.vars["y"].set(round(ann["translation"][1], 3))
        self.vars["z"].set(round(ann["translation"][2], 3))
        self.vars["w"].set(round(ann["size"][0], 3))
        self.vars["l"].set(round(ann["size"][1], 3))
        self.vars["h"].set(round(ann["size"][2], 3))
        self.vars["yaw"].set(round(yaw, 2))

        self._lock_traces = False

        self.render_cameras()

    def on_input_change(self, *args):
        if self._lock_traces or self.selected_ann_idx is None:
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

    def add_annotation(self):
        categories = sorted([cat["name"] for cat in self.nusc.category])
        dialog = CategoryDialog(self.master, categories)
        cat_name = dialog.result

        if not cat_name:
            return

        sample = self.samples[self.current_sample_idx]
        cam_token = sample["data"].get("CAM_FRONT")
        if cam_token:
            cam_data = self.nusc.get("sample_data", cam_token)
            pose_record = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
            start_trans = list(pose_record["translation"])
            start_trans[2] += 1.0
        else:
            start_trans = [0.0, 0.0, 0.0]

        inst_token = uuid.uuid4().hex
        ann_token = uuid.uuid4().hex
        cat_token = next(
            c["token"] for c in self.nusc.category if c["name"] == cat_name
        )

        new_inst = {
            "token": inst_token,
            "category_token": cat_token,
            "nbr_annotations": 1,
            "first_annotation_token": ann_token,
            "last_annotation_token": ann_token,
        }

        vis_token = "4"
        if hasattr(self.nusc, "visibility") and self.nusc.visibility:
            vis_token = self.nusc.visibility[0]["token"]

        new_ann = {
            "token": ann_token,
            "sample_token": sample["token"],
            "instance_token": inst_token,
            "visibility_token": vis_token,
            "attribute_tokens": [],
            "translation": start_trans,
            "size": [2.0, 4.0, 1.5],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "prev": "",
            "next": "",
            "num_lidar_pts": 0,
            "num_radar_pts": 0,
        }

        self.raw_instances.append(new_inst)
        self.raw_inst_map[inst_token] = new_inst
        self.nusc._token2ind["instance"][inst_token] = len(self.raw_instances) - 1

        self.raw_annotations.append(new_ann)
        self.raw_ann_map[ann_token] = new_ann
        self.nusc._token2ind["sample_annotation"][ann_token] = (
            len(self.raw_annotations) - 1
        )

        sample["anns"].append(ann_token)
        self.current_anns.append(new_ann)

        self.ann_listbox.insert(tk.END, f"{cat_name} [{ann_token[:6]}]")
        self.ann_listbox.selection_clear(0, tk.END)
        last_idx = self.ann_listbox.size() - 1
        self.ann_listbox.selection_set(last_idx)

        self.on_ann_select(None)

    def propagate_to_next(self):
        if self.selected_ann_idx is None:
            return

        current_ann = self.current_anns[self.selected_ann_idx]
        current_sample = self.samples[self.current_sample_idx]

        if not current_sample["next"]:
            messagebox.showinfo("Info", "This is the last frame in the scene.")
            return

        next_sample_token = current_sample["next"]
        next_sample_idx = self.nusc._token2ind["sample"][next_sample_token]
        next_sample = self.nusc.sample[next_sample_idx]

        next_ann_token = current_ann.get("next", "")

        if next_ann_token and next_ann_token in self.raw_ann_map:
            next_ann = self.raw_ann_map[next_ann_token]
            next_ann["translation"] = list(current_ann["translation"])
            next_ann["size"] = list(current_ann["size"])
            next_ann["rotation"] = list(current_ann["rotation"])
        else:
            new_ann_token = uuid.uuid4().hex

            current_ann["next"] = new_ann_token
            if current_ann["token"] in self.raw_ann_map:
                self.raw_ann_map[current_ann["token"]]["next"] = new_ann_token

            inst_token = current_ann["instance_token"]

            new_ann = {
                "token": new_ann_token,
                "sample_token": next_sample_token,
                "instance_token": inst_token,
                "visibility_token": current_ann.get("visibility_token", "4"),
                "attribute_tokens": current_ann.get("attribute_tokens", []),
                "translation": list(current_ann["translation"]),
                "size": list(current_ann["size"]),
                "rotation": list(current_ann["rotation"]),
                "prev": current_ann["token"],
                "next": "",
                "num_lidar_pts": 0,
                "num_radar_pts": 0,
            }

            if inst_token in self.raw_inst_map:
                inst = self.raw_inst_map[inst_token]
                inst["nbr_annotations"] += 1
                inst["last_annotation_token"] = new_ann_token

            self.raw_annotations.append(new_ann)
            self.raw_ann_map[new_ann_token] = new_ann

            self.nusc.sample_annotation.append(new_ann)
            self.nusc._token2ind["sample_annotation"][new_ann_token] = (
                len(self.nusc.sample_annotation) - 1
            )
            next_sample["anns"].append(new_ann_token)

        self.save_and_next()

    def mark_start_keyframe(self):
        if self.selected_ann_idx is None:
            messagebox.showinfo("Wait", "Select an annotation first.")
            return
        ann = self.current_anns[self.selected_ann_idx]
        self.keyframe = {
            "sample_idx": self.current_sample_idx,
            "instance_token": ann["instance_token"],
            "translation": np.array(ann["translation"]),
            "size": np.array(ann["size"]),
            "rotation": Quaternion(ann["rotation"]),
            "visibility_token": ann.get("visibility_token", "4"),
            "attribute_tokens": ann.get("attribute_tokens", []),
        }
        self.keyframe_label.config(
            text=f"Keyframe: Frame {self.current_sample_idx}", fg="green"
        )

    def interpolate_to_here(self):
        if self.keyframe is None:
            messagebox.showerror("Error", "Mark a Start Keyframe first!")
            return
        if self.selected_ann_idx is None:
            messagebox.showerror(
                "Error", "Select the end annotation for the track in the current frame."
            )
            return

        end_ann = self.current_anns[self.selected_ann_idx]
        inst_token = end_ann["instance_token"]

        if inst_token != self.keyframe["instance_token"]:
            messagebox.showerror(
                "Error",
                "Instance token mismatch! Ensure you selected the same object track.",
            )
            return

        start_idx = self.keyframe["sample_idx"]
        end_idx = self.current_sample_idx

        if start_idx == end_idx:
            return

        step_dir = 1 if end_idx > start_idx else -1
        total_steps = abs(end_idx - start_idx)

        start_data = self.keyframe
        end_data = {
            "translation": np.array(end_ann["translation"]),
            "size": np.array(end_ann["size"]),
            "rotation": Quaternion(end_ann["rotation"]),
        }

        for step in range(1, total_steps):
            t = step / total_steps
            curr_idx = start_idx + (step * step_dir)
            curr_sample = self.samples[curr_idx]

            interp_trans = (
                start_data["translation"] * (1 - t) + end_data["translation"] * t
            )
            interp_size = start_data["size"] * (1 - t) + end_data["size"] * t
            interp_rot = Quaternion.slerp(
                start_data["rotation"], end_data["rotation"], t
            )

            found_ann_token = None
            for ann_tok in curr_sample["anns"]:
                if (
                    ann_tok in self.raw_ann_map
                    and self.raw_ann_map[ann_tok]["instance_token"] == inst_token
                ):
                    found_ann_token = ann_tok
                    break

            if found_ann_token:
                ann = self.raw_ann_map[found_ann_token]
                ann["translation"] = interp_trans.tolist()
                ann["size"] = interp_size.tolist()
                ann["rotation"] = interp_rot.elements.tolist()
            else:
                new_ann_token = uuid.uuid4().hex
                new_ann = {
                    "token": new_ann_token,
                    "sample_token": curr_sample["token"],
                    "instance_token": inst_token,
                    "visibility_token": start_data["visibility_token"],
                    "attribute_tokens": start_data["attribute_tokens"],
                    "translation": interp_trans.tolist(),
                    "size": interp_size.tolist(),
                    "rotation": interp_rot.elements.tolist(),
                    "prev": "",
                    "next": "",
                    "num_lidar_pts": 0,
                    "num_radar_pts": 0,
                }
                self.raw_annotations.append(new_ann)
                self.raw_ann_map[new_ann_token] = new_ann
                curr_sample["anns"].append(new_ann_token)

        ordered_ann_tokens = []
        for sample in self.samples:
            for ann_tok in sample["anns"]:
                if (
                    ann_tok in self.raw_ann_map
                    and self.raw_ann_map[ann_tok]["instance_token"] == inst_token
                ):
                    ordered_ann_tokens.append(ann_tok)
                    break

        for i, ann_tok in enumerate(ordered_ann_tokens):
            ann = self.raw_ann_map[ann_tok]
            ann["prev"] = ordered_ann_tokens[i - 1] if i > 0 else ""
            ann["next"] = (
                ordered_ann_tokens[i + 1] if i < len(ordered_ann_tokens) - 1 else ""
            )

        if inst_token in self.raw_inst_map:
            inst = self.raw_inst_map[inst_token]
            inst["nbr_annotations"] = len(ordered_ann_tokens)
            if ordered_ann_tokens:
                inst["first_annotation_token"] = ordered_ann_tokens[0]
                inst["last_annotation_token"] = ordered_ann_tokens[-1]

        self.nusc.sample_annotation = self.raw_annotations
        self.nusc._token2ind["sample_annotation"] = {
            a["token"]: idx for idx, a in enumerate(self.raw_annotations)
        }

        self.keyframe = None
        self.keyframe_label.config(text="Keyframe: None", fg="gray")

        messagebox.showinfo(
            "Success", f"Track interpolated & healed across {total_steps} frames!"
        )
        self.render_cameras()

    def delete_annotation(self):
        if self.selected_ann_idx is None:
            return

        if not messagebox.askyesno(
            "Confirm Delete All",
            "Are you sure you want to delete this object across ALL frames?",
        ):
            return

        ann_to_delete = self.current_anns[self.selected_ann_idx]
        inst_tok = ann_to_delete.get("instance_token")

        tokens_to_delete = {
            a["token"]
            for a in self.raw_annotations
            if a.get("instance_token") == inst_tok
        }

        self.raw_annotations = [
            a for a in self.raw_annotations if a["token"] not in tokens_to_delete
        ]
        for t in tokens_to_delete:
            if t in self.raw_ann_map:
                del self.raw_ann_map[t]

        self.nusc.sample_annotation = self.raw_annotations
        self.nusc._token2ind["sample_annotation"] = {
            ann["token"]: i for i, ann in enumerate(self.raw_annotations)
        }

        if inst_tok in self.raw_inst_map:
            del self.raw_inst_map[inst_tok]

        self.raw_instances = [
            i for i in self.raw_instances if i.get("token") != inst_tok
        ]
        self.nusc.instance = self.raw_instances
        self.nusc._token2ind["instance"] = {
            inst["token"]: i for i, inst in enumerate(self.raw_instances)
        }

        for sample in self.nusc.sample:
            sample["anns"] = [t for t in sample["anns"] if t not in tokens_to_delete]

        self.current_anns = [
            a for a in self.current_anns if a["token"] not in tokens_to_delete
        ]

        self.ann_listbox.delete(0, tk.END)
        for ann in self.current_anns:
            cat_name = "unknown"
            try:
                inst = self.nusc.get("instance", ann["instance_token"])
                cat = self.nusc.get("category", inst["category_token"])
                cat_name = cat["name"]
            except Exception:
                pass
            self.ann_listbox.insert(tk.END, f"{cat_name} [{ann['token'][:6]}]")

        self.selected_ann_idx = None
        self._lock_traces = True
        for var in self.vars.values():
            var.set(0.0)
        self._lock_traces = False

        self.render_cameras()

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

            img_resized = cv2.resize(img, (320, 180))
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.cam_labels[cam].config(image=img_tk)
            self.cam_labels[cam].image = img_tk

    def save_and_next(self):
        current_sample = self.samples[self.current_sample_idx]

        # Ensure it is marked as corrected upon saving
        self.corrected_tokens.add(current_sample["token"])

        self.save_progress()
        self.update_progress_ui()
        self.update_status_indicator()

        out_ann_path = os.path.join(self.out_version_dir, "sample_annotation.json")
        out_inst_path = os.path.join(self.out_version_dir, "instance.json")
        try:
            with open(out_ann_path, "w") as f:
                json.dump(self.raw_annotations, f, indent=4)
            with open(out_inst_path, "w") as f:
                json.dump(self.raw_instances, f, indent=4)
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
