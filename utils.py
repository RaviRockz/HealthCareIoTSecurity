import sys
import traceback

import prettytable
from matplotlib import pyplot as plt
from PyQt5.QtCore import (
    QAbstractTableModel,
    QObject,
    QRunnable,
    Qt,
    pyqtSignal,
    pyqtSlot,
)

plt.rcParams["font.family"] = "IBM Plex Mono"

CLASSES = ["Environment", "Patient", "Attack"]


class PandasDfToPyqtTable(QAbstractTableModel):
    def __init__(self, df):
        QAbstractTableModel.__init__(self)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role=None):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._df.columns[col]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return col
        return None


class Stream(QObject):
    fn = pyqtSignal(str)

    def write(self, text):
        self.fn.emit(str(text))

    def flush(self):
        pass


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print(e)
            traceback.print_exc()
            exc_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exc_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


def print_df_to_table(df, p=True):
    field_names = list(df.columns)
    p_table = prettytable.PrettyTable(field_names=field_names)
    p_table.add_rows(df.values.tolist())
    d = "\n".join(
        ["\t\t{0}".format(p_) for p_ in p_table.get_string().splitlines(keepends=False)]
    )
    if p:
        print(d)
    return d
