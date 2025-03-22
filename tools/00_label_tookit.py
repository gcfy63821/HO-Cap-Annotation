import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from pathlib import Path
import shutil
import cv2
import numpy as np
import logging
from time import sleep
from segment_anything import sam_model_registry, SamPredictor
from hocap_annotation.utils import PROJ_ROOT, OBJ_CLASS_COLORS

NUM_COLORS = len(OBJ_CLASS_COLORS)


class ImageLabelToolkit:
    MENU_CHECKABLE = 1
    MENU_DISABLED = 2
    MENU_QUIT = 3

    def __init__(self, device="cuda", debug=False) -> None:
        self._logger = self._init_logger(debug)
        self._device = device
        self._predictor = self._init_sam_predictor()

        self._points = []
        self._undo_stack = []
        self._curr_mask = None
        self._curr_label = 0
        self._masks = []
        self._labels = []
        self._raw_image = None
        self._img_width = 640
        self._img_height = 480
        self._gui_image = o3d.geometry.Image(
            np.zeros((self._img_height, self._img_width, 3), dtype=np.uint8)
        )
        self._text = ""
        self._is_done = False

    def _init_logger(self, debug):
        logger = logging.getLogger("ImageLabelTool")
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _init_sam_predictor(self):
        chkpt = PROJ_ROOT / "config/SAM/sam_vit_l.pth"
        sam = sam_model_registry["vit_l"](checkpoint=chkpt).to(self._device)
        predictor = SamPredictor(sam)
        return predictor

    def run(self):
        self._app = gui.Application.instance
        self._app.initialize()

        # create window
        self._window = self._create_window()

        # add callbacks
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)
        self._window.set_on_key(self._on_key)
        self._widget3d.set_on_mouse(self._on_mouse_widget3d)

        # self._app.run_in_thread(self.update)
        self._app.run()

    def _create_window(self, title="Image Label Tool", width=800, height=600):
        window = gui.Application.instance.create_window(
            title=title, width=width, height=height
        )

        em = window.theme.font_size
        self._panel_width = 20 * em
        margin = 0.25 * em

        self._widget3d = gui.SceneWidget()
        self._widget3d.enable_scene_caching(True)
        self._widget3d.scene = rendering.Open3DScene(window.renderer)
        self._widget3d.scene.set_background(
            [1, 1, 1, 1], o3d.geometry.Image(self._gui_image)
        )
        window.add_child(self._widget3d)

        self._info = gui.Label("")
        self._info.visible = False
        window.add_child(self._info)

        self._panel = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
        # File-chooser widget.
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)
        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Image file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)
        self._panel.add_child(fileedit_layout)
        window.add_child(self._panel)
        # Button "Add Mask", "Remove Mask", "Save Mask"
        button_layout = gui.Horiz(0, gui.Margins(margin, margin, margin, margin))
        addButton = gui.Button("Add Mask")
        removeButton = gui.Button("Remove Mask")
        saveButton = gui.Button("Save Mask")
        addButton.set_on_clicked(self._on_add_mask)
        removeButton.set_on_clicked(self._on_remove_mask)
        saveButton.set_on_clicked(self._on_save_mask)
        button_layout.add_stretch()
        button_layout.add_child(addButton)
        button_layout.add_stretch()
        button_layout.add_child(removeButton)
        button_layout.add_stretch()
        button_layout.add_child(saveButton)
        button_layout.add_stretch()
        self._panel.add_child(button_layout)
        # Mask Label
        blk = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
        blk.add_stretch()
        blk.add_child(gui.Label(f"---Current Mask---"))
        blk.add_stretch()
        blk.add_child(gui.Label(f"Label:"))
        self._intedit = gui.NumberEdit(gui.NumberEdit.INT)
        self._intedit.int_value = 0
        self._intedit.set_on_value_changed(self._on_intedit_changed)
        blk.add_child(self._intedit)
        blk.add_stretch()
        self._panel.add_child(blk)
        # Widget Proxy
        self._proxy = gui.WidgetProxy()
        self._proxy.set_widget(None)
        self._panel.add_child(self._proxy)

        return window

    def _on_intedit_changed(self, value):
        self._curr_label = value

    def _mask_block(self):
        if not self._labels:
            return None
        layout = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        for idx, label in enumerate(self._labels):
            blk = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Mask {idx}:"))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Label: {label}"))
            blk.add_stretch()
            layout.add_child(blk)
        return layout

    def _on_layout(self, ctx):
        pref = self._info.calc_preferred_size(ctx, gui.Widget.Constraints())

        height = self._img_height

        self._widget3d.frame = gui.Rect(0, 0, self._img_width, height)
        self._panel.frame = gui.Rect(
            self._widget3d.frame.get_right(), 0, self._panel_width, height
        )
        self._info.frame = gui.Rect(
            self._widget3d.frame.get_left(),
            self._widget3d.frame.get_bottom() - pref.height,
            pref.width,
            pref.height,
        )
        self._window.size = gui.Size(self._img_width + self._panel_width, height)

    def _on_close(self):
        self._is_done = True
        sleep(0.10)
        return True

    def _on_key(self, event):
        if event.key == gui.KeyName.Q:  # quit
            if event.type == gui.KeyEvent.DOWN:
                self._window.close()
                return True

        if event.key == gui.KeyName.R:  # reset points
            if event.type == gui.KeyEvent.DOWN:
                self._reset()
                return True

        return False

    def _on_add_mask(self):
        self._masks.append(self._curr_mask)
        self._labels.append(self._intedit.int_value)
        print("Labels:", self._labels)
        self._curr_mask = None
        self._curr_label = 0
        self._reset()
        self._proxy.set_widget(self._mask_block())

    def _on_remove_mask(self):
        print("Remove mask")
        self._masks.pop()
        self._labels.pop()
        print("Labels:", self._labels)
        self._proxy.set_widget(self._mask_block())

    def _on_save_mask(self):
        self._make_clean_folder(self._save_folder)
        # save mask
        mask = self.get_mask()
        cv2.imwrite(
            str(self._save_folder / f"{self._iamge_name.replace('color', 'mask')}.png"),
            mask,
        )
        # save mask overlay
        vis_image = cv2.cvtColor(self._overlay_mask(mask), cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(self._save_folder / f"{self._iamge_name.replace('color', 'vis')}.jpg"),
            vis_image,
        )
        print(f"Mask saved to {self._save_folder}")

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self._window.theme)
        filedlg.add_filter(".png .jpg .jepg", "Image files (*.png;*.jpg;*.jpeg)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self._window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self._window.close_dialog()

    def _update_image(self):
        def update_image():
            self._widget3d.scene.set_background(
                [1, 1, 1, 1], o3d.geometry.Image(self._gui_image)
            )
            self._widget3d.force_redraw()

        self._app.post_to_main_thread(self._window, update_image)

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        path = Path(path).resolve()
        if not path.exists():
            return
        self._serial = path.parent.name
        self._iamge_name = path.stem
        self._save_folder = (
            path.parent.parent
            / "data_processing/segmentation/init_segmentation"
            / self._serial
        )
        img = cv2.imread(str(path))
        self._raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._img_height, self._img_width, _ = self._raw_image.shape
        self._gui_image = o3d.geometry.Image(self._raw_image)
        self._predictor.set_image(self._raw_image)
        self._masks = []
        self._labels = []
        self._curr_label = 0
        self._intedit.int_value = 0
        self._curr_mask = None
        self._reset()
        self._proxy.set_widget(self._mask_block())
        self._window.set_needs_layout()
        self._window.close_dialog()

    def _on_mouse_widget3d(self, event):
        if (
            event.type == gui.MouseEvent.Type.BUTTON_DOWN
            and event.is_modifier_down(gui.KeyModifier.CTRL)
            and event.buttons == gui.MouseButton.LEFT.value
        ):
            x = int(event.x - self._widget3d.frame.x)
            y = int(event.y - self._widget3d.frame.y)

            self._points.append((x, y, True))
            self._undo_stack.append(("add", (x, y, True)))
            self._update_sam_mask()
            current_image = self._overlay_mask(self._curr_mask)
            current_image = self._draw_points(current_image, self._points)
            self._gui_image = o3d.geometry.Image(current_image)
            self._update_image()

            return gui.Widget.EventCallbackResult.HANDLED

        if (
            event.type == gui.MouseEvent.Type.BUTTON_DOWN
            and event.is_modifier_down(gui.KeyModifier.CTRL)
            and event.buttons == gui.MouseButton.RIGHT.value
        ):
            x = int(event.x - self._widget3d.frame.x)
            y = int(event.y - self._widget3d.frame.y)

            self._points.append((x, y, False))
            self._undo_stack.append(("add", (x, y, False)))
            self._update_sam_mask()
            current_image = self._overlay_mask(self._curr_mask)
            current_image = self._draw_points(current_image, self._points)
            self._gui_image = o3d.geometry.Image(current_image)
            self._update_image()

            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _reset(self):
        self._points = []
        self._undo_stack = []
        self._gui_image = o3d.geometry.Image(self._raw_image)
        self._update_image()

    def _undo_last_step(self):
        if self._undo_stack:
            action, data = self._undo_stack.pop()
            if action == "add":
                if self._points and self._points[-1] == data:
                    self._points.pop()
            self._update_sam_mask()
            current_image = self._overlay_mask(self._curr_mask)
            current_image = self._draw_points(current_image, self._points)
            self._gui_image = o3d.geometry.Image(current_image)
            self._update_widgetImage()

    def _update_sam_mask(self):
        if self._points:
            # get mask from sam
            input_points = np.array(self._points)[:, :2]
            input_labels = np.array(self._points)[:, 2]
            masks, scores, _ = self._predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            mask = masks[np.argmax(scores)]
            self._curr_mask = mask.astype(np.uint8) * 255
        else:
            self._curr_mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)

    def _overlay_mask(self, mask, alpha=0.6, is_rgb=True):
        unique_labels = np.unique(mask)
        if len(unique_labels) == 1:
            return self._raw_image
        unique_labels = unique_labels[unique_labels != 0]  # Removing background
        # Map each object label in the mask to a color
        overlay = np.zeros_like(self._raw_image)
        for label in unique_labels:
            mask_color = OBJ_CLASS_COLORS[(label - 1) % NUM_COLORS]
            if not is_rgb:
                mask_color = mask_color[::-1]
            overlay[mask == label] = mask_color
        # Blend the color image and the mask
        blended = cv2.addWeighted(self._raw_image, 1 - alpha, overlay, alpha, 0)
        return blended

    def _draw_points(self, img, points, is_rgb=True):
        img_copy = img.copy()
        for x, y, label in points:
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            if is_rgb:
                color = color[::-1]
            cv2.circle(img_copy, (x, y), 3, color, -1)
        return img_copy

    def _make_clean_folder(self, folder):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    def get_mask(self):
        mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)
        if self._masks:
            for idx, m in enumerate(self._masks):
                mask[m == 255] = self._labels[idx]
        else:
            mask[self._curr_mask == 255] = self._curr_label
        return mask


if __name__ == "__main__":
    toolkit = ImageLabelToolkit()
    toolkit.run()
