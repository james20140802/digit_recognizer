"""Module for creating gui."""
import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSlot
import pyqtgraph
from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtCore


class Main(QWidget):
    def __init__(self):
        super().__init__()

        # Define variables for saving metrics.
        self.epoch = []
        self.train_accuracy = []
        self.train_loss = []
        self.validation_accuracy = []
        self.validation_loss = []

        # Create layout.
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()

        vbox = QVBoxLayout()

        group_box1 = QVBoxLayout()
        group_box2 = QVBoxLayout()
        group_box3 = QVBoxLayout()
        group_box4 = QVBoxLayout()
        group_box5 = QVBoxLayout()

        # Set default font.
        font_tick = QtGui.QFont("Bahnschrift SemiLight", 8)
        font = QtGui.QFont("Bahnschrift SemiLight", 12)
        font.setBold(True)
        self.setFont(font)

        # Generate 4 graphs and set x axis type string.
        self.stringaxis_train_accuracy = pyqtgraph.AxisItem(orientation="bottom")
        self.stringaxis_train_loss = pyqtgraph.AxisItem(orientation="bottom")
        self.stringaxis_validation_accuracy = pyqtgraph.AxisItem(orientation="bottom")
        self.stringaxis_validation_loss = pyqtgraph.AxisItem(orientation="bottom")

        self.train_accuracy_graph = pyqtgraph.PlotWidget(
            axisItems={"bottom": self.stringaxis_train_accuracy}
        )
        self.train_loss_graph = pyqtgraph.PlotWidget(
            axisItems={"bottom": self.stringaxis_train_loss}
        )
        self.validation_accuracy_graph = pyqtgraph.PlotWidget(
            axisItems={"bottom": self.stringaxis_validation_accuracy}
        )
        self.validation_loss_graph = pyqtgraph.PlotWidget(
            axisItems={"bottom": self.stringaxis_validation_loss}
        )

        # Generate graph title.
        self.train_accuracy_graph.setTitle("Train Accuracy", color="#828282", size="12pt")
        self.train_loss_graph.setTitle("Train Loss", color="#828282", size="12pt")
        self.validation_accuracy_graph.setTitle("Validation Accuracy", color="#828282", size="12pt")
        self.validation_loss_graph.setTitle("Validation Loss", color="#828282", size="12pt")

        # Set graph title font.
        self.train_accuracy_graph.getPlotItem().titleLabel.item.setFont(font)
        self.train_loss_graph.getPlotItem().titleLabel.item.setFont(font)
        self.validation_accuracy_graph.getPlotItem().titleLabel.item.setFont(font)
        self.validation_loss_graph.getPlotItem().titleLabel.item.setFont(font)

        # Set background color of gaphs.
        self.train_accuracy_graph.setBackground((240, 240, 240))
        self.train_loss_graph.setBackground((240, 240, 240))
        self.validation_accuracy_graph.setBackground((240, 240, 240))
        self.validation_loss_graph.setBackground((240, 240, 240))

        # Set graph pen.
        self.train_accuracy_curve = self.train_accuracy_graph.plot(
            pen=pyqtgraph.mkPen(color=(203, 26, 126), width=3, style=QtCore.Qt.SolidLine)
        )
        self.train_loss_curve = self.train_loss_graph.plot(
            pen=pyqtgraph.mkPen(color=(44, 106, 180), width=3, style=QtCore.Qt.DotLine)
        )
        self.validation_accuracy_curve = self.validation_accuracy_graph.plot(
            pen=pyqtgraph.mkPen(color=(145, 122, 184), width=4, style=QtCore.Qt.SolidLine)
        )
        self.validation_loss_curve = self.validation_loss_graph.plot(
            pen=pyqtgraph.mkPen(color=(203, 26, 126), width=3, style=QtCore.Qt.SolidLine)
        )

        # Set style for name of axis.
        label_style = {"color": "#828282", "font-size": "9pt"}

        # Set name for axis.
        self.train_accuracy_graph.setLabel("left", "Train Accuracy", **label_style)
        self.train_loss_graph.setLabel("left", "Train Loss", **label_style)
        self.validation_accuracy_graph.setLabel("left", "Validation Accuracy", **label_style)
        self.validation_loss_graph.setLabel("left", "Validation Loss", **label_style)
        self.train_accuracy_graph.setLabel("bottom", "Epoch", **label_style)
        self.train_loss_graph.setLabel("bottom", "Epoch", **label_style)
        self.validation_accuracy_graph.setLabel("bottom", "Epoch", **label_style)
        self.validation_loss_graph.setLabel("bottom", "Epoch", **label_style)

        # Set font and gradation of axis.
        self.train_accuracy_graph.getAxis("bottom").setStyle(tickFont=font_tick, tickTextOffset=6)
        self.train_loss_graph.getAxis("bottom").setStyle(tickFont=font_tick, tickTextOffset=6)
        self.validation_accuracy_graph.getAxis("bottom").setStyle(
            tickFont=font_tick, tickTextOffset=6
        )
        self.validation_loss_graph.getAxis("bottom").setStyle(tickFont=font_tick, tickTextOffset=6)
        self.train_accuracy_graph.getAxis("left").setStyle(tickFont=font_tick, tickTextOffset=6)
        self.train_loss_graph.getAxis("left").setStyle(tickFont=font_tick, tickTextOffset=6)
        self.validation_accuracy_graph.getAxis("left").setStyle(
            tickFont=font_tick, tickTextOffset=6
        )
        self.validation_loss_graph.getAxis("left").setStyle(tickFont=font_tick, tickTextOffset=6)

        # Generate Data Indicator Group Box.
        self.groupbox_epoch = QGroupBox("Epoch")
        self.groupbox_train_accuracy = QGroupBox("Train Accuracy")
        self.groupbox_train_loss = QGroupBox("Train Loss")
        self.groupbox_validation_accuracy = QGroupBox("Validation Accuracy")
        self.groupbox_validation_loss = QGroupBox("Validation Loss")

        # Generate Data Indicator Label.
        self.label_epoch = QLabel("0", self)
        self.label_train_accuracy = QLabel("0", self)
        self.label_train_loss = QLabel("0", self)
        self.label_validation_accuracy = QLabel("0", self)
        self.label_validation_loss = QLabel("0", self)

        # Arrange Data Indicator central.
        self.label_epoch.setAlignment(Qt.AlignCenter)
        self.label_train_accuracy.setAlignment(Qt.AlignCenter)
        self.label_train_loss.setAlignment(Qt.AlignCenter)
        self.label_validation_accuracy.setAlignment(Qt.AlignCenter)
        self.label_validation_loss.setAlignment(Qt.AlignCenter)

        # Set background color and border of Data Indicator.
        self.label_epoch.setStyleSheet(
            "color:rgb(0, 0, 0);"
            "background-color:rgb(250,250,250);"
            "border-style: solid;"
            "border-width: 1px;"
            "border-color: rgb(200,200,200);"
            "border-radius: 5px"
        )
        self.label_train_accuracy.setStyleSheet(
            "color:rgb(203, 26, 126);"
            "background-color:rgb(250,250,250);"
            "border-style: solid;"
            "border-width: 1px;"
            "border-color: rgb(200,200,200);"
            "border-radius: 5px"
        )
        self.label_train_loss.setStyleSheet(
            "color:rgb(203, 26, 126);"
            "background-color:rgb(250,250,250);"
            "border-style: solid;"
            "border-width: 1px;"
            "border-color: rgb(200,200,200);"
            "border-radius: 5px"
        )
        self.label_validation_accuracy.setStyleSheet(
            "color:rgb(44, 106, 180);"
            "background-color:rgb(250,250,250);"
            "border-style: solid;"
            "border-width: 1px;"
            "border-color: rgb(200,200,200);"
            "border-radius: 5px"
        )
        self.label_validation_loss.setStyleSheet(
            "color:rgb(44, 106, 180);"
            "background-color:rgb(250,250,250);"
            "border-style: solid;"
            "border-width: 1px;"
            "border-color: rgb(200,200,200);"
            "border-radius: 5px"
        )

        # Vertically locate Group Box and Label.
        group_box1.addWidget(self.label_epoch)
        group_box2.addWidget(self.label_train_accuracy)
        group_box3.addWidget(self.label_train_loss)
        group_box4.addWidget(self.label_validation_accuracy)
        group_box5.addWidget(self.label_validation_loss)

        self.groupbox_epoch.setLayout(group_box1)
        self.groupbox_train_accuracy.setLayout(group_box2)
        self.groupbox_train_loss.setLayout(group_box3)
        self.groupbox_validation_accuracy.setLayout(group_box4)
        self.groupbox_validation_loss.setLayout(group_box5)

        # Group horizontally. (1 row: 2 Graphs), (2 row : 2 Graphs), (3 row : 10 Labels(Data Indicator))
        hbox1.addWidget(self.train_accuracy_graph)
        hbox1.addWidget(self.train_loss_graph)

        hbox2.addWidget(self.validation_accuracy_graph)
        hbox2.addWidget(self.validation_loss_graph)

        hbox3.addWidget(self.groupbox_epoch)
        hbox3.addWidget(self.groupbox_train_accuracy)
        hbox3.addWidget(self.groupbox_train_loss)
        hbox3.addWidget(self.groupbox_validation_accuracy)
        hbox3.addWidget(self.groupbox_validation_loss)

        # Group grouped widgets vertically.
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)

        # Create window and arrange layout.
        self.setLayout(vbox)
        self.setGeometry(100, 100, 1300, 500)
        self.setWindowTitle("Accuracy & Loss Monitoring")

        # Make range of axis x.
        self.train_accuracy_graph.enableAutoRange(axis="x")
        self.train_loss_graph.enableAutoRange(axis="x")
        self.validation_accuracy_graph.enableAutoRange(axis="x")
        self.validation_loss_graph.enableAutoRange(axis="x")

        # Make range of axis y.
        self.train_accuracy_graph.enableAutoRange(axis="y")
        self.train_loss_graph.enableAutoRange(axis="y")
        self.validation_accuracy_graph.enableAutoRange(axis="y")
        self.validation_loss_graph.enableAutoRange(axis="y")

        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.update)
        self.timer.start()

        self.show()

    @pyqtSlot()
    def update(self):
        self.train_accuracy_curve.setData(self.epoch, self.train_accuracy)
        self.train_loss_curve.setData(self.epoch, self.train_loss)
        self.validation_accuracy_curve.setData(self.epoch, self.validation_accuracy)
        self.validation_loss_curve.setData(self.epoch, self.validation_loss)

        self.label_epoch.setText(str(self.epoch[-1]))
        self.label_train_accuracy.setText(str(self.train_accuracy[-1]))
        self.label_train_loss.setText(str(self.train_loss[-1]))
        self.label_validation_accuracy.setText(str(self.validation_accuracy[-1]))
        self.label_validation_loss.setText(str(self.validation_loss[-1]))

    def update_metrics(
        self, epoch, train_accuracy, train_loss, validation_accuracy, validation_loss
    ):
        self.epoch = epoch
        self.train_accuracy = train_accuracy
        self.train_loss = train_loss
        self.validation_accuracy = validation_accuracy
        self.validation_loss = validation_loss


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Main()
    sys.exit(app.exec_())
