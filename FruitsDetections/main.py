# This Python file uses the following encoding: utf-8
import sys
import os


from PySide6.QtWidgets import QApplication, QWidget,QFileDialog
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

class FruitsDetection(QWidget):
    def __init__(self):
        super(FruitsDetection, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load("form.ui")
        self.ui.show()
        self.ui.btn_address.clicked.connect(self.getfile)
        self.file=''

    def getfile(self):
        self.fileName = QFileDialog.getOpenFileName(self,'Single File','C:\'','*.jpg *.mp4 *.jpeg *.png *.avi')
        self.file=str(self.fileName[0])
        self.ui.txt_address.setText(self.file)
        return self.file
        



