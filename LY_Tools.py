# coding=gbk

from __init__ import *

class Tools:
    '''
    小工具
    '''

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Tools\\'
        self.mk_dir(self.this_class_arr, force=True)
        pass

    def mk_dir(self, dir, force=False):

        if not os.path.isdir(dir):
            if force == True:
                os.makedirs(dir)
            else:
                os.mkdir(dir)

    def load_npy(self,f):

        return dict(np.load(f).item())


    def load_df(self,f):
        df = pd.read_pickle(f)
        df = pd.DataFrame(df)
        return df
        pass

    def save_df(self,df,outf):
        if outf.endswith('.df'):
            df.to_pickle(outf)
        else:
            df.to_pickle(outf + '.df')


    def mask_999999_arr(self,arr):
        arr[arr<-99999]=np.nan

    def lonlat_to_address(self,lon, lat):
        ak = "mziulWyNDGkBdDnFxWDTvELlMSun8Obt"  # 参照自己的应用
        url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak=mziulWyNDGkBdDnFxWDTvELlMSun8Obt&output=json&coordtype=wgs84ll&location=%s,%s' % (
        lat, lon)
        content = requests.get(url).text
        dic = eval(content)
        # for key in dic['result']:
        add = dic['result']['formatted_address']
        return add


    def spatial_arr_filter_n_sigma(self,spatial_arr,n=3):
        arr_std = np.nanstd(spatial_arr)
        arr_mean = np.nanmean(spatial_arr)
        top = arr_mean + n*arr_std
        bottom = arr_mean - n*arr_std
        spatial_arr[spatial_arr>top] = np.nan
        spatial_arr[spatial_arr<bottom] = np.nan




    def pix_to_address(self, pix):
        # 只适用于单个像素查看，不可大量for循环pix，存在磁盘重复读写现象
        outf = self.this_class_arr + 'pix_to_address_history.npy'
        if not os.path.isfile(outf):
            np.save(outf, {0: 0})
        pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr+'pix_to_lon_lat_dic.npy'
        if not os.path.isfile(pix_to_lon_lat_dic_f):
            DIC_and_TIF().spatial_tif_to_lon_lat_dic()
        lon_lat_dic = self.load_npy(pix_to_lon_lat_dic_f)
        # print(pix)
        lon, lat = lon_lat_dic[pix]
        print(lon, lat)

        history_dic = self.load_npy(outf)

        if pix in history_dic:
            # print(history_dic[pix])
            return lon, lat, history_dic[pix]
        else:
            address = self.lonlat_to_address(lon, lat).decode('utf-8')
            key = pix
            val = address
            history_dic[key] = val
            np.save(outf, history_dic)
            return lon, lat, address


    def interp_1d(self, val,threashold):
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= threashold:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        # if flag == 0:
        #     return
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)

        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = range(len(val))
        yiii = interp_1(xiii)

        return yiii

    def interp_1d_1(self, val,threshold):
        # 不插离群值 只插缺失值
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= threshold:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)


        return yi



    def interp_nan(self,val,kind='nearest'):
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if not np.isnan(val[i]):
                flag += 1.
                index = i
                x.append(index)
                # val_new = np.append(val_new, val[i])
                val_new.append(val[i])
        if flag / len(val) < 0.3:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind=kind, fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)


        return yi

        pass


    def detrend_dic(self, dic):
        dic_new = {}
        for key in dic:
            vals = dic[key]
            if len(vals) == 0:
                dic_new[key] = []
                continue
            vals_new = signal.detrend(vals)
            dic_new[key] = vals_new

        return dic_new

    def arr_mean(self, arr, threshold):
        grid = arr > threshold
        arr_mean = np.mean(arr[np.logical_not(grid)])
        return arr_mean

    def arr_mean_nan(self,arr):

        flag = 0.
        sum_ = 0.
        x = []
        for i in arr:
            if np.isnan(i):
                continue
            sum_ += i
            flag += 1
            x.append(i)
        if flag == 0:
            return np.nan,np.nan
        else:
            mean = sum_/flag
            # xerr = mean/np.std(x,ddof=1)
            xerr = np.std(x)
            # print mean,xerr
            # if xerr > 10:
            #     print x
            #     print xerr
            #     print '........'
            #     plt.hist(x,bins=10)
            #     plt.show()
            #     exit()
            return mean,xerr

    def pick_vals_from_2darray(self, array, index):
        # 2d
        ################# check zone #################
        # plt.imshow(array)
        # for r,c in index:
        #     # print(r,c)
        #     array[r,c] = 100
        # #     # exit()
        # plt.figure()
        # plt.imshow(array)
        # plt.show()
        ################# check zone #################
        picked_val = []
        for r, c in index:
            val = array[r, c]
            if np.isnan(val):
                continue
            picked_val.append(val)
        picked_val = np.array(picked_val)
        return picked_val
        pass

    def pick_vals_from_1darray(self, arr, index):
        # 1d
        picked_vals = []
        for i in index:
            picked_vals.append(arr[i])
        picked_vals = np.array(picked_vals)
        return picked_vals

    def pick_min_indx_from_1darray(self, arr, indexs):
        min_index = 99999
        min_val = 99999
        # plt.plot(arr)
        # plt.show()
        for i in indexs:
            val = arr[i]
            # print val
            if val < min_val:
                min_val = val
                min_index = i
        return min_index

    def pick_max_indx_from_1darray(self, arr, indexs):
        max_index = 99999
        max_val = -99999
        # plt.plot(arr)
        # plt.show()
        for i in indexs:
            val = arr[i]
            # print val
            if val > max_val:
                max_val = val
                max_index = i
        return max_index


    def point_to_shp(self, inputlist, outSHPfn):
        '''

        :param inputlist:

        # input list format
        # [[lon,lat,val],
        #      ...,
        # [lon,lat,val]]

        :param outSHPfn:
        :return:
        '''

        if len(inputlist) > 0:
            outSHPfn = outSHPfn + '.shp'
            fieldType = ogr.OFTReal
            # Create the output shapefile
            shpDriver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(outSHPfn):
                shpDriver.DeleteDataSource(outSHPfn)
            outDataSource = shpDriver.CreateDataSource(outSHPfn)
            outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)
            idField1 = ogr.FieldDefn('val', fieldType)
            outLayer.CreateField(idField1)
            for i in range(len(inputlist)):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(inputlist[i][0], inputlist[i][1])
                featureDefn = outLayer.GetLayerDefn()
                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(point)
                outFeature.SetField('val', inputlist[i][2])
                # 加坐标系
                spatialRef = osr.SpatialReference()
                spatialRef.ImportFromEPSG(4326)
                spatialRef.MorphToESRI()
                file = open(outSHPfn[:-4] + '.prj', 'w')
                file.write(spatialRef.ExportToWkt())
                file.close()

                outLayer.CreateFeature(outFeature)
                outFeature.Destroy()
            outFeature = None


class SMOOTH:
    '''
    一些平滑算法
    '''

    def __init__(self):

        pass

    def interp_1d(self, val):
        if len(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= -10:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.9:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)

        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma
        # plt.scatter(range(len(yi)),yi)
        # print(len(yi),123)
        # plt.scatter(range(len(yi)),yi)
        # plt.plot(yi)
        # plt.show()
        # print(len(yi))

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = range(len(val))
        yiii = interp_1(xiii)

        # for i in range(len(yi)):
        #     if yi[i] == -999999:
        #         val_new_ii = np.append(val_new_ii, bottom)
        #     elif yi[i] == 999999:
        #         val_new_ii = np.append(val_new_ii, top)
        #     else:
        #         val_new_ii = np.append(val_new_ii, yi[i])

        return yiii

    def smooth_convolve(self, x, window_len=11, window='hanning'):
        """
        1d卷积滤波
        smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.
        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        # return y
        return y[(window_len / 2 - 1):-(window_len / 2)]

    def smooth(self, x):
        # 后窗滤波
        # 滑动平均
        x = np.array(x)
        temp = 0
        new_x = []
        for i in range(len(x)):
            if i + 3 == len(x):
                break
            temp += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
            new_x.append(temp / 4.)
            temp = 0
        return np.array(new_x)


    def smooth_interpolate(self,inx,iny,zoom):
        '''
        1d平滑差值
        :param inlist:
        :return:
        '''

        x_new = np.arange(min(inx),max(inx),((max(inx)-min(inx))/float(len(inx)))/float(zoom))
        func = interpolate.interp1d(inx,iny,kind='cubic')
        y_new = func(x_new)
        return x_new,y_new

    def forward_window_smooth(self, x, window=3):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        for i in range(len(x)):
            if i - window < 0:
                new_x = np.append(new_x, x[i])
            else:
                temp = 0
                for w in range(window):
                    temp += x[i - w]
                smoothed = temp / float(window)
                new_x = np.append(new_x, smoothed)
        return new_x

    def filter_3_sigma(self, arr_list):
        sum_ = []
        for i in arr_list:
            if i >= 0:
                sum_.append(i)
        sum_ = np.array(sum_)
        val_mean = np.mean(sum_)
        sigma = np.std(sum_)
        n = 3
        sum_[(val_mean - n * sigma) > sum_] = -999999
        sum_[(val_mean + n * sigma) < sum_] = -999999

        # for i in
        return sum_

        pass


    def hist_plot_smooth(self,arr,interpolate_window=5,**kwargs):
        weights = np.ones_like(arr) / float(len(arr))
        n1, x1, patch = plt.hist(arr,weights=weights,**kwargs)
        density1 = stats.gaussian_kde(arr)
        y1 = density1(x1)
        coe = max(n1) / max(y1)
        y1 = y1 * coe
        x1, y1 = self.smooth_interpolate(x1, y1, interpolate_window)
        return x1,y1

        pass


class DIC_and_TIF:
    '''
    字典转tif
    tif转字典
    '''

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\DIC_and_TIF\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        self.tif_template = this_root + 'conf\\tif_template.tif'
        pass


    def arr_to_tif(self, array, newRasterfn):
        # template
        tif_template = self.tif_template
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = -999999
        to_raster.array2raster(newRasterfn, originX, originY, pixelWidth, pixelHeight, array)
        pass

    def arr_to_tif_GDT_Byte(self, array, newRasterfn):
        # template
        tif_template = self.tif_template
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = 255
        to_raster.array2raster_GDT_Byte(newRasterfn, originX, originY, pixelWidth, pixelHeight, array)
        pass


    def spatial_arr_to_dic(self,arr):

        pix_dic = {}
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                pix = (i,j)
                val = arr[i][j]
                pix_dic[pix] = val

        return pix_dic


    def pix_dic_to_spatial_arr(self, spatial_dic):

        # x = []
        # y = []
        # for key in spatial_dic:
        #     key_split = key.split('.')
        #     x.append(key_split[0])
        #     y.append(key_split[1])
        # row = len(set(x))
        # col = len(set(y))
        tif_template = self.tif_template
        arr_template, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        row = len(arr_template)
        col = len(arr_template[0])
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        # hist = []
        # for v in all_vals:
        #     if not np.isnan(v):
        #         if 00<v<1.5:
        #             hist.append(v)

        spatial = np.array(spatial,dtype=float)
        return spatial

    def pix_dic_to_spatial_arr_mean(self, spatial_dic):

        mean_spatial_dic = {}
        for pix in tqdm(spatial_dic,desc='calculating spatial mean'):
            vals = spatial_dic[pix]
            if len(vals) == 0:
                mean = np.nan
            else:
                mean = np.nanmean(vals)
            mean_spatial_dic[pix] = mean

        spatial = self.pix_dic_to_spatial_arr(mean_spatial_dic)
        spatial = np.array(spatial,dtype=float)
        return spatial


    def pix_dic_to_spatial_arr_ascii(self, spatial_dic):
        # dtype can be in ascii format
        # x = []
        # y = []
        # for key in spatial_dic:
        #     key_split = key.split('.')
        #     x.append(key_split[0])
        #     y.append(key_split[1])
        # row = len(set(x))
        # col = len(set(y))
        tif_template = self.tif_template
        arr_template, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        row = len(arr_template)
        col = len(arr_template[0])
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        spatial = np.array(spatial)
        return spatial


    def pix_dic_to_tif(self, spatial_dic, out_tif):

        spatial = self.pix_dic_to_spatial_arr(spatial_dic)
        # spatial = np.array(spatial)
        self.arr_to_tif(spatial, out_tif)

    def spatial_tif_to_lon_lat_dic(self):
        tif_template = self.tif_template
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        # print(originX, originY, pixelWidth, pixelHeight)
        # exit()
        pix_to_lon_lat_dic = {}
        for i in tqdm(range(len(arr))):
            for j in range(len(arr[0])):
                pix = (i, j)
                lon = originX + pixelWidth * j
                lat = originY + pixelHeight * i
                pix_to_lon_lat_dic[pix] = [lon, lat]
        print('saving')
        np.save(self.this_class_arr + 'pix_to_lon_lat_dic', pix_to_lon_lat_dic)


    def spatial_tif_to_dic(self,tif):

        arr = to_raster.raster2array(tif)[0]
        Tools().mask_999999_arr(arr)
        dic = self.spatial_arr_to_dic(arr)
        return dic

        pass

    def void_spatial_dic(self):
        tif_template = self.tif_template
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = (row, col)
                void_dic[key] = []
        return void_dic


    def void_spatial_dic_nan(self):
        tif_template = self.tif_template
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = (row, col)
                void_dic[key] = np.nan
        return void_dic

    def void_spatial_dic_zero(self):
        tif_template = self.tif_template
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = (row, col)
                void_dic[key] = 0.
        return void_dic

    def plot_back_ground_arr(self):
        tif_template = self.tif_template
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        back_ground = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val < -90000:
                    temp.append(np.nan)
                else:
                    temp.append(1)
            back_ground.append(temp)
        back_ground = np.array(back_ground)
        plt.imshow(back_ground, 'gray', vmin=0, vmax=1.4)

        # return back_ground

        pass


    def ascii_to_arr(self,lonlist,latlist,vals):
        '''
        transform ascii text to spatial array
        :param lonlist:[.....]
        :param latlist: [.....]
        :param vals: [.....]
        :return:
        :todo: need to be modified
        '''
        # matrix = np.meshgrid(lonlist,latlist)
        lons = list(set(lonlist))
        lats = list(set(latlist))
        print lons
        lons.sort()
        lats.sort()
        lon_matri,lat_matri = np.meshgrid(lons,lats)
        # print matrix
        # for i in range(len(lonlist)):
        #     print type(lonlist[i]),latlist[i],vals[i]
        #     sleep()

        # lon_lat_dic = dict(np.load(self.this_class_arr + 'pix_to_lon_lat_dic.npy').item())
        # lon_lat_dic_reverse = {}
        # for key in lon_lat_dic:
        #     lon,lat = lon_lat_dic[key]
        #     new_key = str(lon)+'_'+str(lat)
        #     print new_key
        #     sleep()
        #     lon_lat_dic_reverse[new_key] = key

        # spatial_dic = {}
        # for i in range(len(lonlist)):
        #     lt = str(lonlist[i])+'_'+str(latlist[i])
        #     pix = lon_lat_dic_reverse[lt]
        #     spatial_dic[pix] = vals[i]

        # arr = self.pix_dic_to_spatial_arr_ascii(spatial_dic)
        # return arr
        exit()


    def mask_ocean_dic(self):
        tif_template = self.tif_template
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        ocean_dic = {}
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val < -99999:
                    continue
                else:
                    ocean_dic[(i,j)]=1
        return ocean_dic


class MULTIPROCESS:
    '''
    可对类内的函数进行多进程并行
    由于GIL，多线程无法跑满CPU，对于不占用CPU的计算函数可用多线程
    并行计算加入进度条
    '''

    def __init__(self, func, params):
        self.func = func
        self.params = params
        copy_reg.pickle(types.MethodType, self._pickle_method)
        pass

    def _pickle_method(self, m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    def run(self, process=-9999, process_or_thread='p', **kwargs):
        '''
        # 并行计算加进度条
        :param func: input a kenel_function
        :param params: para1,para2,para3... = params
        :param process: number of cpu
        :param thread_or_process: multi-thread or multi-process,'p' or 't'
        :param kwargs: tqdm kwargs
        :return:
        '''

        if process > 0:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool(process)
            elif process_or_thread == 't':
                pool = TPool(process)
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results
        else:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool()
            elif process_or_thread == 't':
                pool = TPool()
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results


class KDE_plot:

    def __init__(self):

        pass

    def reverse_colourmap(self, cmap, name='my_cmap_r'):
        """
        In:
        cmap, name
        Out:
        my_cmap_r
        Explanation:
        t[0] goes from 0 to 1
        row i:   x  y0  y1 -> t[0] t[1] t[2]
                       /
                      /
        row i+1: x  y0  y1 -> t[n] t[1] t[2]
        so the inverse should do the same:
        row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                       /
                      /
        row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
        """
        reverse = []
        k = []

        for key in cmap._segmentdata:
            k.append(key)
            channel = cmap._segmentdata[key]
            data = []

            for t in channel:
                data.append((1 - t[0], t[2], t[1]))
            reverse.append(sorted(data))

        LinearL = dict(zip(k, reverse))
        my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
        return my_cmap_r

    def makeColours(self, vals, cmap, reverse=0):
        norm = []
        for i in vals:
            norm.append((i - np.min(vals)) / (np.max(vals) - np.min(vals)))
        colors = []
        cmap = plt.get_cmap(cmap)
        if reverse:
            cmap = self.reverse_colourmap(cmap)
        else:
            cmap = cmap

        for i in norm:
            colors.append(cmap(i))
        return colors

    def linefit(self,x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx,sy,sxx,syy,sxy=0,0,0,0,0
        for i in range(0,int(N)):
            sx  += x[i]
            sy  += y[i]
            sxx += x[i]*x[i]
            syy += y[i]*y[i]
            sxy += x[i]*y[i]
        a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
        b = (sy - a*sx)/N
        r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
        return a,b,r


    def plot_fit_line(self,a,b,r,X,Y,title='',**argvs):
        '''
        画拟合直线 y=ax+b
        画散点图 X,Y
        :param a:
        :param b:
        :param X:
        :param Y:
        :param i:
        :param title:
        :return:
        '''
        x = np.linspace(min(X),max(X),10)
        y = a*x + b
        #
        # plt.subplot(2,2,i)
        # plt.scatter(X,Y,marker='o',s=5,c = 'grey')
        # plt.plot(X,Y)
        if not 'linewidth' in argvs:
            plt.plot(x, y, linestyle='dashed', c='black', linewidth=1, alpha=0.7,label='y={:0.2f}x+{:0.2f}\nr={:0.2f}'.format(a,b,r), **argvs)
        else:
            plt.plot(x,y,linestyle='dashed',c='black',alpha=0.7,label='y={:0.2f}x+{:0.2f}\nr={:0.2f}'.format(a,b,r),**argvs)
        plt.title(title)


    def plot_scatter(self, val1, val2,plot_fit_line=False,is_plot_1_1_line=False, cmap='magma', reverse=0, s=0.3, title='',ax=None,**kwargs):
        val1 = np.array(val1)
        val2 = np.array(val2)
        print 'data length is {}'.format(len(val1))
        if len(val1) > 30000:
            val_range_index = range(len(val1))
            val_range_index = random.sample(val_range_index, 30000)  # 从val中随机选择30000个点，目的是加快核密度算法
            new_val1 = []
            new_val2 = []
            for i in val_range_index:
                new_val1.append(val1[i])
                new_val2.append(val2[i])
            val1 = new_val1
            val2 = new_val2
            print 'data length is modified to {}'.format(len(val1))
        else:
            val1 = val1
            val2 = val2

        kde_val = np.array([val1, val2])
        print('doing kernel density estimation... ')
        densObj = kde(kde_val)
        dens_vals = densObj.evaluate(kde_val)
        colors = self.makeColours(dens_vals, cmap, reverse=reverse)
        if ax == None:
            plt.figure()
            plt.title(title)
            plt.scatter(val1, val2, c=colors, s=s,**kwargs)
        else:
            plt.title(title)
            plt.scatter(val1, val2, c=colors, s=s,**kwargs)
        if plot_fit_line:
            a, b, r = self.linefit(val1,val2)
            if is_plot_1_1_line:
                plt.plot([np.min([val1,val2]), np.max([val1,val2])], [np.min([val1,val2]), np.max([val1,val2])], '--', c='black')
            self.plot_fit_line(a,b,r,val1,val2)
            plt.legend()
            return a,b,r

class Pre_Process:

    def __init__(self):
        pass

    def run(self):

        pass

    def do_data_transform(self):
        father_dir = this_root + 'SPEI\\tif\\'
        for spei_dir in os.listdir(father_dir):
            print spei_dir + '\n'
            interval = spei_dir[-2:]

            spei_dir_ = spei_dir.upper()[:4] + '_' + interval
            outdir = this_root + 'SPEI\\per_pix\\' + spei_dir_ + '\\'
            print outdir
            Tools().mk_dir(outdir)
            self.data_transform(father_dir + spei_dir + '\\', outdir)


    def kenel_data_transfrom(self,params):

        col,all_array,r,void_dic = params



        pass

    def data_transform(self, fdir, outdir):
        # 不可并行，内存不足
        Tools().mk_dir(outdir)
        # 将空间图转换为数组
        # per_pix_data
        flist = os.listdir(fdir)
        date_list = []
        for f in flist:
            if f.endswith('.tif'):
                date = f.split('.')[0]
                date_list.append(date)
        date_list.sort()
        all_array = []
        for d in tqdm(date_list, 'loading...'):
            # for d in date_list:
            for f in flist:
                if f.endswith('.tif'):
                    if f.split('.')[0] == d:
                        # print(d)
                        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
                        array = np.array(array,dtype=np.float)
                        # print np.min(array)
                        # print type(array)
                        # plt.imshow(array)
                        # plt.show()
                        all_array.append(array)

        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic[(r, c)] = []
                void_dic_list.append((r, c))

        # print(len(void_dic))
        # exit()
        params = []
        for r in tqdm(range(row)):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic[(r, c)].append(val)

        # for i in void_dic_list:
        #     print(i)
        # exit()
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            temp_dic[key] = void_dic[key]
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def kernel_cal_anomaly(self, params):
        fdir, f, save_dir = params
        pix_dic = dict(np.load(fdir + f).item())
        anomaly_pix_dic = {}
        for pix in pix_dic:
            ####### one pix #######
            vals = pix_dic[pix]
            # 清洗数据
            climatology_means = []
            climatology_std = []
            # vals = signal.detrend(vals)
            for m in range(1, 13):
                one_mon = []
                for i in range(len(pix_dic[pix])):
                    mon = i % 12 + 1
                    if mon == m:
                        one_mon.append(pix_dic[pix][i])
                mean = np.nanmean(one_mon)
                std = np.nanstd(one_mon)
                climatology_means.append(mean)
                climatology_std.append(std)

            # 算法1
            # pix_anomaly = {}
            # for m in range(1, 13):
            #     for i in range(len(pix_dic[pix])):
            #         mon = i % 12 + 1
            #         if mon == m:
            #             this_mon_mean_val = climatology_means[mon - 1]
            #             this_mon_std_val = climatology_std[mon - 1]
            #             if this_mon_std_val == 0:
            #                 anomaly = -999999
            #             else:
            #                 anomaly = (pix_dic[pix][i] - this_mon_mean_val) / float(this_mon_std_val)
            #             key_anomaly = i
            #             pix_anomaly[key_anomaly] = anomaly
            # arr = pandas.Series(pix_anomaly)
            # anomaly_list = arr.to_list()
            # anomaly_pix_dic[pix] = anomaly_list

            # 算法2
            pix_anomaly = []
            for i in range(len(vals)):
                mon = i % 12
                std_ = climatology_std[mon]
                mean_ = climatology_means[mon]
                if std_ == 0:
                    anomaly = 0 ##### 修改gpp
                else:
                    anomaly = (vals[i] - mean_) / std_

                pix_anomaly.append(anomaly)
            # pix_anomaly = Tools().interp_1d_1(pix_anomaly,-100)
            # plt.plot(pix_anomaly)
            # plt.show()
            anomaly_pix_dic[pix] = pix_anomaly

        np.save(save_dir + f, anomaly_pix_dic)

    def cal_anomaly(self,fdir,save_dir):
        # fdir = this_root + 'NDVI\\per_pix\\'
        # save_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        Tools().mk_dir(save_dir)
        flist = os.listdir(fdir)
        # flag = 0
        params = []
        for f in flist:
            # print(f)
            params.append([fdir, f, save_dir])

        # for p in params:
        #     print(p[1])
        #     self.kernel_cal_anomaly(p)
        MULTIPROCESS(self.kernel_cal_anomaly, params).run(process=6, process_or_thread='p',
                                                         desc='calculating anomaly...')


    def smooth_anomaly(self):
        fdir = this_root+'NDVI\\per_pix_anomaly\\'
        outdir = this_root+'NDVI\\per_pix_anomaly_smooth\\'
        Tools().mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            dic = dict(np.load(fdir+f).item())
            smooth_dic = {}
            for key in dic:
                vals = dic[key]
                smooth_vals = SMOOTH().forward_window_smooth(vals)
                smooth_dic[key] = smooth_vals
            np.save(outdir+f,smooth_dic)



    def check_ndvi_anomaly(self):
        fdir = this_root + 'NDVI\\per_pix\\'
        for f in os.listdir(fdir):
            dic = dict(np.load(fdir+f).item())

            for pix in tqdm(dic):
                val = dic[pix]
                std = np.std(val)
                if std == 0 or len(val) == 0:
                    continue
                # print val
                val = Tools().interp_1d_1(val,-3000)
                # print val
                if len(val) == 1:
                    continue
                plt.plot(val)
                plt.grid()
                plt.show()
        pass

    def check_per_pix(self,fdir):

        for f in os.listdir(fdir):
            dic = dict(np.load(fdir+f).item())
            for pix in dic:
                val = dic[pix]
                print pix,val

    def extend_GPP(self):
        fidr = this_root + 'GPP\\per_pix_anomaly\\'
        outdir = this_root + 'GPP\\per_pix_anomaly_extend\\'
        Tools().mk_dir(outdir)
        for f in tqdm(os.listdir(fidr)):
            # if not '015' in f:
            #     continue
            dic = dict(np.load(fidr + f).item())
            new_dic = {}
            for key in dic:
                val = dic[key]
                n = len(val)
                if n == 0:
                    new_dic[key] = []
                    continue
                ni = 408 - 192
                null_list = [np.nan] * ni
                null_list.extend(val)
                new_dic[key] = null_list
            np.save(outdir + f, new_dic)


def main():
    raise UserWarning('Do not run this script')
    pass


if __name__ == '__main__':
    main()