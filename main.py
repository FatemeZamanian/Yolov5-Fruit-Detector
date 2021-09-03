import sys
import os

from PySide6.QtWidgets import QApplication, QWidget,QFileDialog
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PyQt5.QtGui import QImage
from PySide6.QtGui import QPixmap
import cv2
from PySide6.QtCore import QThread, Signal, QDir


def convertCVImage2QtImage(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    qimg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class show_video(QThread):
    signal_show_frame = Signal(object)

    def __init__(self):
        super(show_video, self).__init__()

    def run(self):
        self.video = cv2.VideoCapture(0)

        while True:
            valid, self.frame = self.video.read()
            if valid is not True:
                break
            self.signal_show_frame.emit(self.frame)
            cv2.waitKey(30)

    def stop(self):
        try:
            self.video.release()
        except:
            pass




class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load("ui/form.ui")
        self.ui.show()
        self.ui.btn_browse.clicked.connect(self.getfile)
        self.ui.btn_start.clicked.connect(self.getfile)
        self.file=''

    def getfile(self):
        self.fileName = QFileDialog.getOpenFileName(self,'Single File','C:\'','*.jpg *.mp4 *.jpeg *.png *.avi')
        self.file = self.fileName[0]
        self.ui.txt_address.setText(str(self.file))
        
        self.ui.lbl_out.setPixmap(self.file)
        return self.file
        
    def predict(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    sys.exit(app.exec())
