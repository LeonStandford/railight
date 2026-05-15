from typing import Any
from ultralytics.solutions.solutions import (
    BaseSolution,
    SolutionAnnotator,
    SolutionResults,
)
from ultralytics.utils.plotting import colors


class VisionEye(BaseSolution):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vision_point = self.CFG["vision_point"]

    def process(self, im0) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)
        for cls, t_id, box, conf in zip(
            self.clss, self.track_ids, self.boxes, self.confs
        ):
            annotator.box_label(
                box,
                label=self.adjust_box_label(cls, conf, t_id),
                color=colors(int(t_id), True),
            )
            annotator.visioneye(box, self.vision_point)
        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
