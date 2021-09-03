import sys
import os

from PySide6.QtWidgets import QApplication, QWidget,QFileDialog
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

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
        self.file = str(self.fileName[0])
        self.ui.txt_address.setText(self.file)
        return self.file
        
    def predict(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    sys.exit(app.exec())
