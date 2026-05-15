from typing import Any
from ultralytics.solutions.solutions import (
    BaseSolution,
    SolutionAnnotator,
    SolutionResults,
)
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class SecurityAlarm(BaseSolution):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email: str, password: str, to_email: str) -> None:
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com", 587)
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records: int = 5) -> None:
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        import cv2

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "Security Alert"
        message_body = f"Ultralytics alert: {records} object(s) detected."
        message.attach(MIMEText(message_body))
        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")
        message.attach(image_attachment)
        try:
            self.server.send_message(message)
            LOGGER.info("Email sent successfully!")
        except Exception as e:
            LOGGER.error(f"Failed to send email: {e}")

    def process(self, im0) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        for box, cls in zip(self.boxes, self.clss):
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
        total_det = len(self.clss)
        if total_det >= self.records and (not self.email_sent) and self.server:
            self.send_email(im0, total_det)
            self.email_sent = True
        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(
            plot_im=plot_im,
            total_tracks=len(self.track_ids),
            email_sent=self.email_sent,
        )
