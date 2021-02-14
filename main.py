import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWebEngineWidgets
from PyQt5 import uic
import pandas as pd
import gzip
import pickle
import folium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import csv
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QBrush
from sklearn.impute import KNNImputer
from tqdm import tqdm
import impyute
import random
from resnet1d import resnet50_1d
import argparse
import errno

pd.set_option('display.max_colwidth', None)
form_class = uic.loadUiType("untitled1.ui")[0]
form_class1 = uic.loadUiType("untitled.ui")[0]


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data
        self.colors = dict()

    def rowCount(self, parent=None):
        return self._data.index.size

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
            if role == Qt.EditRole:
                return str(self._data.iloc[index.row(), index.column()])
            if role == Qt.BackgroundRole:
                color = self.colors.get((index.row(), index.column()))
                if color is not None:
                    return color
        return None

    def headerData(self, rowcol, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[rowcol]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return self._data.index[rowcol]
        return None

    def flags(self, index):
        flags = super(self.__class__, self).flags(index)
        flags |= Qt.ItemIsEditable
        flags |= Qt.ItemIsSelectable
        flags |= Qt.ItemIsEnabled
        flags |= Qt.ItemIsDragEnabled
        flags |= Qt.ItemIsDropEnabled
        return flags

    def sort(self, Ncol, order):
        """Sort table by given column number.
        """
        try:
            self.layoutAboutToBeChanged.emit()
            self._data = self._data.sort_values(
                self._data.columns[Ncol], ascending=not order
            )
            self.layoutChanged.emit()
        except Exception as e:
            print(e)

    def change_color(self, row, column, color):
        ix = self.index(row, column)
        self.colors[(row, column)] = color
        self.dataChanged.emit(ix, ix, (Qt.BackgroundRole,))


class NewWindow(QMainWindow, form_class1):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.fileName = ""
        self.ta = pd.DataFrame(columns=['MDATETIME','AREA_INDEX', 'RAIN', 'SO2',
                                        'NO2', 'O3', 'CO', 'PM10', 'PM25', 'NO', 'NOX'])


class WindowClass(QMainWindow, form_class):
    def __init__(self, fileName):
        super().__init__()
        self.setupUi(self)
        self.fileName = ""
        self.df1 = []  # 최초 csv파일 dataframe(설정 변환됨)
        self.df2 = []  # 최초 csv파일 dataframe(설정 그대로)
        self.df3 = {}  # 선택한 일자 측정소와 data dict
        self.df4 = []  # 선택한 일자에 측정한 측정소 번호 list
        self.df5 = pd.DataFrame()  # graph출력위해 해당 선택한 측정소에 해당하는 dataframe
        self.index = []  # 전체 측정소 list
        self.df6 = pd.DataFrame(columns=['MDATETIME','AREA_INDEX', 'RAIN', 'SO2',
                                         'NO2', 'O3', 'CO', 'PM10', 'PM25', 'NO', 'NOX'])  # 새로운 창에 갈 데이터
        self.df7={}     #1440데이터 측정소별 저장
        self.indexnum={}
        self.sym_bools=[]
        self.air_bools=[]

        # print(model)
        # print(type(model))

        center = [35.541, 126.986]
        m = folium.Map(location=center, zoom_start=6)
        data = io.BytesIO()
        m.save(data, close_file=False)
        self.webEngineView.setHtml(data.getvalue().decode())

        # 불러오기
        self.import_2.clicked.connect(self.file_open)
        self.datetime.pressed.connect(self.combo)
        self.locselection.pressed.connect(self.settext)
        self.refilebutton.pressed.connect(self.refile)
        self.pushButton.pressed.connect(self.showgraph)
        self.analysisButton.pressed.connect(self.make_input)
        self.new_window = NewWindow(self)
        self.showdfButton.pressed.connect(self.showdf)

    # 불러오기 함수
    def file_open(self):
        path, _ = QFileDialog.getOpenFileName(self)

        with gzip.open(path, 'rb') as f:
            self.df1 = pickle.load(f)
        self.df2 = self.df1.copy()
        self.df3.clear()
        self.df4.clear()
        # print("open")

        # list up area_index
        self.index = []
        for key in self.df1.keys():
            self.index.append(key)
        # print(self.index)

        data = self.df1[self.index[0]]['MDATETIME']
        data.drop_duplicates(keep='first', inplace=True)
        data.reset_index(inplace=True, drop=True)
        # print(data)

        for i in range(0, len(data)):
            self.comboBox.addItem(str(data[i]))
            self.comboBox_3.addItem(str(data[i]))
            self.comboBox_4.addItem(str(data[i]))

    def combo(self):
        content = self.comboBox.currentText()  # 선택한 날짜

        # # print(self.df2)
        # iscontent = self.df2['YYYYMMDDHH'].astype(str)==content
        # self.df3 = self.df2[iscontent]
        # self.df3.drop_duplicates(keep='first', inplace = True)
        # print(self.df3)
        for i in range(0, len(self.index)):
            iscontent = self.df2[self.index[i]]['MDATETIME'] == content
            self.df3[self.index[i]] = self.df2[self.index[i]][iscontent]

        for i in range(0, len(self.index)):
            if (content in self.df2[self.index[i]]['MDATETIME'].values):
                self.df4.append(self.index[i])
        iloc = pd.read_excel('loc.xlsx')

        center = [35.541, 126.986]
        m = folium.Map(location=center, zoom_start=6)

        for i in tqdm(range(0, len(self.df4))):
            title = self.df4[i]
            self.comboBox_2.addItem(str(self.df4[i]))

            for f in range(0, len(iloc)):
                if self.df4[i] == iloc.loc[f, '측정소\n코드']:
                    sub_lat = iloc.loc[f, '위도']
                    sub_long = iloc.loc[f, '경도']
                    folium.Marker([sub_lat, sub_long], tooltip=title, popup="normal").add_to(m)
                    # print(self.df2[self.df3[i]][self.df2[self.df3[i]]['MDATETIME']==content])
                    # print(type(self.df3[list(self.df3)[i]]['SO2']))
                    # if float(self.df2[self.df3[i]][self.df2[self.df3[i]]['MDATETIME']==content]) > 0 and float(self.df3[list(self.df3)[i]]['SO2']) < 1:
                    #     folium.Marker([sub_lat,sub_long],tooltip = title,popup="normal").add_to(m)
                    # else:
                    #     folium.Marker([sub_lat,sub_long],tooltip = title, icon=folium.Icon(color='red',icon='star'),popup="SO2 error").add_to(m)

        data = io.BytesIO()
        m.save(data, close_file=False)
        self.webEngineView.setHtml(data.getvalue().decode())

    def settext(self):
        station_num = self.comboBox_2.currentText()
        data1 = self.df3[int(station_num)].transpose()
        data = data1.reset_index()
        model = pandasModel(data)
        self.tableView.setModel(model)

    def refile(self):
        # self.knn_imputation()
        self.showmsgbox()

    def knn_imputation(self):
        impu_df = self.df2.copy()
        for area in impu_df.keys():
            for col in ['SO2', 'NO2', 'NO', 'NOX', 'CO', 'O3', 'PM10', 'PM25']:
                flag = col + '_FLAG'
                idx = impu_df[area][impu_df[area][col] == 999999].index
                impu_df[area][col][idx] = float("nan")
                impu_df[area][flag][idx] = 0

                idx = impu_df[area][impu_df[area][flag] == 0].index
                impu_df[area][col][idx] = float("nan")

        colume = ['SO2', 'NO2', 'NO', 'NOX', 'CO', 'O3', 'PM10', 'PM25']

        def knn(item_list, data):
            print(item_list, 'start')
            f = open('./dataset/log.txt', 'a')
            key_list = list(np.append(item_list, ['AREA_INDEX', 'TIME_INDEX']))
            for item in item_list:
                key_list.append(item + '_FLAG')

            normal = {}
            for key in data.keys():
                normal[key] = data[key][key_list]

                for item in item_list:
                    flag = item + '_FLAG'
                    normal[key] = normal[key][normal[key][flag] == 1]

            for key in normal.keys():
                normal[key] = normal[key][item_list]
                normal[key].index = range(normal[key][item_list[0]].size)

            for key in tqdm(normal.keys()):
                if normal[key][item_list[0]].size == 0:
                    f.write('err : ' + str(item_list) + ' ' + str(key) + ' ' + str(
                        np.sum(data[key][item_list[0]].isna())) + '\n')
                    continue
                np_normal = normal[key].to_numpy()
                imputer = KNNImputer(n_neighbors=2)
                imputer.fit(np_normal)
                test = data[key][item_list].to_numpy()
                try:
                    test = imputer.transform(test)
                except:
                    f.write('err : ' + str(item_list) + ' ' + str(key) + ' ' + str(
                        np.sum(data[key][item_list[0]].isna())) + '\n')
                    continue
                data[key][item_list] = pd.DataFrame(test, columns=item_list)

            f.close()
            return data

        def knn_time(item_list, data):
            # i_l : ['PM10', 'PM25', 'NO', 'NO2', 'O3', 'SO2', 'CO']
            # data: df2
            # k_l : ['PM10', 'PM25', 'NO', 'NO2', 'O3', 'SO2', 'CO', 'AREA_INDEX', 'TIME_INDEX', 'PM10_FLAG', 'PM25_FLAG', 'NO_FLAG', 'NO2_FLAG', 'O3_FLAG', 'SO2_FLAG', 'CO_FLAG']
            f = open('./dataset/log.txt', 'a')
            key_list = list(np.append(item_list, ['AREA_INDEX', 'TIME_INDEX']))
            for item in item_list:
                key_list.append(item + '_FLAG')

            normal = {}
            for key in data.keys():
                normal[key] = data[key][key_list]

                for item in item_list:
                    flag = item + '_FLAG'
                    normal[key] = normal[key][normal[key][flag] == 1]
                # print("normal["+str(key)+"]")
                # print(normal[key])
            tmp_list = item_list
            tmp_list.append('TIME_INDEX')

            for key in normal.keys():
                tmp_list = item_list
                tmp_list.append('TIME_INDEX')
                normal[key] = normal[key][tmp_list]
                normal[key].index = range(normal[key][tmp_list[0]].size)

            # for key in data.keys():
            #     for item in item_list:
            #         index = data[key][data[key][item+'_FLAG'] == 0].index
            #         data[key][item][index] = np.nan

            for key in tqdm(normal.keys()):
                if normal[key][tmp_list[0]].size == 0:
                    f.write('err : ' + str(tmp_list) + ' ' + str(key) + ' ' + str(
                        np.sum(data[key][tmp_list[0]].isna())) + '\n')
                    continue
                np_normal = normal[key].to_numpy()
                imputer = KNNImputer(n_neighbors=2)
                imputer.fit(np_normal)

                test = data[key][tmp_list].to_numpy()
                try:
                    test = imputer.transform(test)
                except:
                    f.write('err : ' + str(tmp_list) + ' ' + str(key) + ' ' + str(
                        np.sum(data[key][tmp_list[0]].isna())) + '\n')
                    continue
                data[key][tmp_list] = pd.DataFrame(test, columns=tmp_list)

            f.close()
            return data

        result = impu_df.copy()

        attr_list = [['PM10', 'PM25', 'NO', 'NO2', 'O3', 'SO2', 'CO']]
        for key in result.keys():
            result[key].index = range(len(result[key]))

        for attr in attr_list:
            print("check 22222222")
            print(attr)
            result = knn_time(attr, result)

        del_idx = [324148, 336133, 437132, 437151, 131120, 131198, 131414, 131194]
        for id in del_idx:
            del result[id]

        del_idx2 = list()
        for area in result.keys():
            if len(result[area]) < 1440:
                del_idx2.append(area)

        for id in del_idx2:
            del result[id]

        for area in result.keys():
            flag = 'NOX_FLAG'
            idx = result[area][result[area][flag] == 0].index
            result[area]['NOX'][idx] = (result[area]['NO'][idx] + result[area]['NO2'][idx])

        for area in result.keys():
            for col in ['SO2', 'NO2', 'NO', 'CO', 'O3', 'PM10', 'PM25']:
                flag = col + '_FLAG'
                idx = result[area][result[area][flag] == 0].index
                result[area][flag][idx] = 1

        with gzip.open('./dataset/knn_impu_time.pickle', 'wb') as f:
            pickle.dump(result, f)

    def analysis(self):
        self.combo()
        self.settext()
        self.showgraph()

    def showmsgbox(self):
        msgBox = QMessageBox()
        msgBox.setWindowTitle("complete")
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("완료되었습니다.")
        msgBox.setStandardButtons(QMessageBox.Cancel)
        msgBox.setDefaultButton(QMessageBox.Yes)
        msgBox.exec_()

    # def errorpoint(self):
    def showgraph(self):
        start_time = str(self.comboBox_3.currentText())
        end_time = str(self.comboBox_4.currentText())
        station_num = int(self.comboBox_2.currentText())
        time_x = []
        SO2_y = []
        CO_y = []
        NO_y = []
        NO2_y = []
        NOX_y = []
        O3_y = []
        PM10_y = []
        PM25_y = []

        self.df5 = self.df2[station_num].copy()
        self.df5.reset_index(drop=True, inplace=True)
        date_num = 1
        for i in tqdm(range(0, len(self.df5))):
            # print(self.df2[station_num])
            if str(self.df5.loc[i, 'MDATETIME']) >= start_time and str(self.df5.loc[i, 'MDATETIME']) <= end_time:
                time_x.append(date_num)
                SO2_y.append(float(self.df5.loc[i, 'SO2']))
                CO_y.append(float(self.df5.loc[i, 'CO']))
                NO_y.append(float(self.df5.loc[i, 'NO']))
                NO2_y.append(float(self.df5.loc[i, 'NO2']))
                NOX_y.append(float(self.df5.loc[i, 'NOX']))
                O3_y.append(float(self.df5.loc[i, 'O3']))
                PM10_y.append(float(self.df5.loc[i, 'PM10']))
                PM25_y.append(float(self.df5.loc[i, 'PM25']))
                date_num += 1

        self.SO2.canvas.axes.cla()
        self.SO2.canvas.axes.plot(time_x, SO2_y, label='SO2')
        self.SO2.canvas.axes.legend()
        self.SO2.canvas.axes.set_xlabel('time')
        self.SO2.canvas.axes.set_ylabel('SO2')
        self.SO2.canvas.draw()

        self.CO.canvas.axes.cla()
        self.CO.canvas.axes.plot(time_x, CO_y, label='CO')
        self.CO.canvas.axes.legend()
        self.CO.canvas.axes.set_xlabel('time')
        self.CO.canvas.axes.set_ylabel('CO')
        self.CO.canvas.draw()

        self.NO.canvas.axes.cla()
        self.NO.canvas.axes.plot(time_x, NO_y, label='NO')
        self.NO.canvas.axes.legend()
        self.NO.canvas.axes.set_ylabel('NO')
        self.NO.canvas.axes.set_xlabel('time')
        self.NO.canvas.draw()

        self.NO2.canvas.axes.cla()
        self.NO2.canvas.axes.plot(time_x, NO2_y, label='NO2')
        self.NO2.canvas.axes.legend()
        self.NO2.canvas.axes.set_ylabel('NO2')
        self.NO2.canvas.axes.set_xlabel('time')
        self.NO2.canvas.draw()

        self.NOX.canvas.axes.cla()
        self.NOX.canvas.axes.plot(time_x, NOX_y, label='NOX')
        self.NOX.canvas.axes.legend()
        self.NOX.canvas.axes.set_ylabel('NOX')
        self.NOX.canvas.axes.set_xlabel('time')
        self.NOX.canvas.draw()

        self.O3.canvas.axes.cla()
        self.O3.canvas.axes.plot(time_x, O3_y, label='O3')
        self.O3.canvas.axes.legend()
        self.O3.canvas.axes.set_ylabel('O3')
        self.O3.canvas.axes.set_xlabel('time')
        self.O3.canvas.draw()

        self.PM10.canvas.axes.cla()
        self.PM10.canvas.axes.plot(time_x, PM10_y, label='PM10')
        self.PM10.canvas.axes.legend()
        self.PM10.canvas.axes.set_ylabel('PM10')
        self.PM10.canvas.axes.set_xlabel('time')
        self.PM10.canvas.draw()

        self.PM25.canvas.axes.cla()
        self.PM25.canvas.axes.plot(time_x, PM25_y, label='PM2.5')
        self.PM25.canvas.axes.legend()
        self.PM25.canvas.axes.set_ylabel('PM2.5')
        self.PM25.canvas.axes.set_xlabel('time')
        self.PM25.canvas.draw()

    def make_input(self):

        test_loss = 0.0
        true_air_labels = []
        pred_air_labels = []

        true_sym_labels = []
        pred_sym_labels = []

        model = resnet50_1d(n_sym=6, n_air=9).cuda()
        model.load_state_dict(torch.load("resnet1d-2990-regular.pth"))
        model.eval()
        transform = transforms.Compose([transforms.ToTensor()])
        with gzip.open('./raw_data_MH/knn_impu_4.pickle', 'rb') as f:
            checkpk = pickle.load(f)
        check_date = self.comboBox.currentText()

        def preProcess3(_data: pd.DataFrame, _window_size: int = 1440):
            """[summary]

            Args:
                _data (pd.DataFrame): [description]
                _window_size (int, optional): [description]. Defaults to 1440.

            Returns:
                [dict]:  [train_data]: data_value, ["train_label"]: element_label (last index: 1: normal) , ["label_attr"] : FLAG
            """
            data = _data.iloc[0:_window_size]
            data_input = data[['RAIN', 'SO2', 'NO2', 'NO', 'NOX', 'O3', 'CO', 'PM10', 'PM25']]
            data_flag = data[['SO2_FLAG', 'NO2_FLAG', 'NO_FLAG', 'NOX_FLAG', 'O3_FLAG', 'CO_FLAG', 'PM10_FLAG',
                              'PM25_FLAG']].to_numpy()

            empty_dict = {'train_data': [], 'train_label': [], 'label_attr': []}

            attr_label = np.zeros(6)
            label = np.zeros(9)

            for idx in range(_window_size // 2, _window_size):
                tmp_attr = np.zeros(6)
                tmp_label = np.zeros(9)
                check_sum = 0

                for c in range(8):
                    val = data_flag[idx][c]
                    if val != 1:
                        tmp_label[c] = 1

                        if val == 3:
                            tmp_attr[1] = 1

                        elif val == 5:
                            tmp_attr[2] = 1

                        elif val == 7:
                            tmp_attr[3] = 1

                        elif val == 10:
                            tmp_attr[4] = 1

                        elif val == 11:
                            tmp_attr[5] = 1

                    elif val == 1:
                        tmp_attr[0] = 1
                        check_sum += 1

                if check_sum == 8:
                    tmp_label[8] = 1

                label = np.vstack((label, tmp_label))
                attr_label = np.vstack((attr_label, tmp_attr))

            empty_dict['train_label'] = np.where(label[1:].sum(axis=0) > 0, 1, 0)
            empty_dict['label_attr'] = np.where(attr_label[1:].sum(axis=0) > 0, 1, 0)
            empty_dict['train_data'] = data_input.to_numpy()
            return empty_dict
        idnum=0

        for idx in self.index:
            # print("check : " + str(idx))
            try:
                newdf = checkpk[idx].copy()
                newdf.reset_index(inplace=True, drop=True)
            except KeyError as ex:
                print("key error!! ||||  " + str(ex))
                continue

            date_index = -1
            for i in range(0, len(newdf)):
                if newdf['MDATETIME'][i] == check_date:
                    date_index = i
                    break

            if date_index < 1440:
                print(str(idx) + "번 측정소에는 " + str(check_date) + "부터 2달치 데이터가 없습니다.")
                continue


            cpy_df = newdf.iloc[date_index - 1440:date_index, :].copy()
            cpy_df.reset_index(inplace=True, drop=True)
            dt = preProcess3(cpy_df)
            self.df7[int(idx)] = cpy_df
            self.indexnum[int(idx)]=idnum
            idnum = idnum + 1

            air_label = torch.tensor([dt['train_label']])
            sym_label = torch.tensor([dt['label_attr']])
            input_air = dt['train_data']

            input_air = transform(input_air)
            input_air = input_air.permute(0, 2, 1)

            images = Variable(input_air.float().cuda())
            air_label = Variable(air_label.float().cuda())
            sym_label = Variable(sym_label.float().cuda())

            outputs = model(images)
            # print(outputs)

            loss_function = nn.BCEWithLogitsLoss()
            air_loss = loss_function(outputs['air'], air_label)  # loss는 점수. 많이 틀릴수록 높은 점수
            sym_loss = loss_function(outputs['sym'], sym_label)

            test_loss += (air_loss.item() + sym_loss.item()) / 2

            # gpu에 있던 값들을 cpu로 옮기고 numpy로 변경
            air_label = air_label.detach().cpu().numpy()
            true_air_labels.append(air_label)

            pred_air_label = torch.sigmoid(outputs['air'])
            pred_air_label = pred_air_label.to('cpu').detach().numpy()
            pred_air_labels.append(pred_air_label)

            sym_label = sym_label.detach().cpu().numpy()

            true_sym_labels.append(sym_label)
            pred_sym_label = torch.sigmoid(outputs['sym'])
            pred_sym_label = pred_sym_label.to('cpu').detach().numpy()
            pred_sym_labels.append(pred_sym_label)

        pred_air_labels = [item for sublist in pred_air_labels for item in sublist]
        pred_sym_labels = [item for sublist in pred_sym_labels for item in sublist]
        pred_air_bools=[]
        for pl in pred_air_labels:
            n_pl = [pl[0]<0.9745 , pl[1]<0.9777, pl[2]<0.9765, pl[3]<0.9930, pl[4]<0.9780, pl[5]<0.9750, pl[6]>0.9350, pl[7]>0.9358, pl[8]<0.9974]
            pred_air_bools.append(n_pl)
        pred_sym_bools = []
        for pl in pred_sym_labels:
            n_pl=[pl[0]<0.998 , pl[1]<0.988 , pl[2]>0.1755 , pl[3]>0.1757, pl[4]>0.1750, pl[5]>0.780]
            pred_sym_bools.append(n_pl)

        # print("pred_air_labels:")
        # print(pred_air_labels)
        # print("pred_sym_labels:")
        # print(pred_sym_labels)
        print("pred_air_bools:")
        print(pred_air_bools)
        print("pred_sym_bools:")
        print(pred_sym_bools)

        self.sym_bools=pred_sym_bools
        self.air_bools=pred_air_bools
        self.showmsgbox()

    def showdf(self):
        station_num = int(self.comboBox_2.currentText())
        ndf = pd.DataFrame(columns=['MDATETIME','AREA_INDEX', 'RAIN', 'SO2',
                                         'NO2', 'O3', 'CO', 'PM10', 'PM25', 'NO', 'NOX'])  # 새로운 창에 갈 데이터
        # for i in (0,len(self.df7[station_num])):
        #     ndf = ndf.append({'AREA_INDEX' : station_num, 'RAIN' : self.df7[station_num].loc[i,'RAIN'],
        #             'SO2' : self.df7[station_num].loc[i,'SO2'],'NO2':self.df7[station_num].loc[i,'NO2'],'O3' :self.df7[station_num].loc[i,'O3'],'CO': self.df7[station_num].loc[i,'CO'],
        #             'PM10': self.df7[station_num].loc[i,'PM10'],'PM25': self.df7[station_num].loc[i,'PM25'],'NO': self.df7[station_num].loc[i,'NO'],'NOX': self.df7[station_num].loc[i,'NOX']}, ignore_index=True)

        ndf=self.df7[station_num][['MDATETIME','AREA_INDEX', 'RAIN','SO2', 'NO2','O3','CO','PM10','PM25','NO','NOX']]
        # print(ndf)
        color=self.air_bools[self.indexnum[station_num]]
        strin = 'flag : '
        if(self.sym_bools[self.indexnum[station_num]][1]==1) : strin = strin + '3, '
        if(self.sym_bools[self.indexnum[station_num]][2]==1) : strin = strin + '5, '
        if(self.sym_bools[self.indexnum[station_num]][3]==1) : strin = strin + '7, '
        if(self.sym_bools[self.indexnum[station_num]][4]==1) : strin = strin + '10, '
        if(self.sym_bools[self.indexnum[station_num]][5]==1) : strin = strin + '11, '
        m = pandasModel(ndf)
        for i in range(0,1440) :
            for j in range(0,len(color)):
                if(color[j]==False) :
                    m.change_color(i, j+2, QBrush(Qt.red))
        self.new_window.tableView.setModel(m)
        self.new_window.textEdit.setText(strin)
        self.new_window.show()





app = QApplication(sys.argv)
mainWindow = WindowClass('')
mainWindow.show()
app.exec_()
