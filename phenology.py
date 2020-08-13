# coding=gbk
from __init__ import *
from HANTS import *

class Phenology:

    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Phenology\\'
        self.this_class_tif = results_root + 'tif\\Phenology\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):
        # 1 把多年的NDVI分成单年，分南北半球，分文件夹存储
        # fdir = data_root+'NDVI_phenology\\tif_05deg_bi_weekly\\'
        # outdir = data_root+'NDVI_phenology\\tif_05deg_bi_weekly_separate\\'
        # self.split_north_south_hemi(fdir,outdir)

        # 2 把单年的NDVI tif 转换成 perpix
        # for folder in ['north','south_modified']:
        #     fdir = data_root+'NDVI_phenology\\tif_05deg_bi_weekly_separate\\{}\\'.format(folder)
        #     outdir = data_root+'NDVI_phenology\\per_pix_separate\\{}\\'.format(folder)
        #     self.data_transform_split_files(fdir,outdir)
        # 3 hants smooth
        # self.hants()
        self.check_hants()

        # 4 计算 top left right
        # self.SOS_EOS()
        # self.check_SOS_EOS()

        # 5 合成南北半球
        # self.compose_SOS_EOS()
        # pass


    def compose_SOS_EOS(self):
        outdir = self.this_class_arr + 'compose_SOS_EOS\\'
        T.mk_dir(outdir)
        threshold_i = 0.5
        SOS_EOS_dic = DIC_and_TIF().void_spatial_dic()
        for hemi in ['north','south_modified']:
            fdir = self.this_class_arr + 'SOS_EOS\\threshold_{}\\{}\\'.format(threshold_i, hemi)
            for f in os.listdir(fdir):
                # year = f.split('.')[0]
                # year = int(year)
                print fdir + f
                dic = T.load_npy(fdir + f)
                for pix in dic:
                    vals = dic[pix]
                    vals = np.array(vals)
                    a,b = pix.split('.')
                    a = int(a)
                    b = int(b)
                    pix = (a,b)
                    SOS_EOS_dic[pix].append(vals)
        SOS_EOS_dic_np = {}
        for pix in SOS_EOS_dic:
            val = SOS_EOS_dic[pix]
            val = np.array(val)
            SOS_EOS_dic_np[pix] = val
        np.save(outdir + 'compose_SOS_EOS',SOS_EOS_dic_np)
        # spatial_dic = {}
        # for pix in SOS_EOS_dic:
        #     vals = SOS_EOS_dic[pix]
        #     length = len(vals)
        #     spatial_dic[pix] = length
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()
            # exit()
        pass

    def split_north_south_hemi(self,fdir,outdir):
        # 1 north
        north_dir = outdir + 'north\\'
        south_dir = outdir + 'south\\'
        T.mk_dir(south_dir,force=True)
        T.mk_dir(north_dir,force=True)
        years = np.array(range(1982, 2016))
        for y in tqdm(years):
            for f in os.listdir(fdir):
                if not f.endswith('tif'):
                    continue
                year = int(f[:4])
                if y == year:
                    date = f[4:8]
                    # print date
                    # exit()
                    array,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir+f)
                    array_south = copy.copy(array)
                    array_north = copy.copy(array)
                    array_south[:180] = -999999.
                    array_north[180:] = -999999.
                    north_outdir_y = north_dir + '{}\\'.format(y)
                    south_outdir_y = south_dir + '{}\\'.format(y)
                    Tools().mk_dir(north_outdir_y)
                    Tools().mk_dir(south_outdir_y)
                    south_outf = south_outdir_y + date + '.tif'
                    north_outf = north_outdir_y + date + '.tif'
                    DIC_and_TIF().arr_to_tif(array_south,south_outf)
                    DIC_and_TIF().arr_to_tif(array_north,north_outf)
        # 2 modify south
        south_modified_dir = outdir + 'south_modified\\'
        T.mk_dir(south_modified_dir)
        for year in os.listdir(south_dir):
            outdir_y = south_modified_dir + '{}\\'.format(year)
            T.mk_dir(outdir_y)
            for f in os.listdir(south_dir + year):
                mon = f.split('.')[0][:2]
                day = f.split('.')[0][2:]
                year = int(year)
                mon = int(mon)
                # print 'original date:',year,mon
                old_fname = south_dir + str(year) + '\\' + f
                mon = mon - 6
                if mon <= 0:
                    year_new = year - 1
                    mon_new = mon + 12
                    new_fname = south_modified_dir + '{}\\{:02d}{}.tif'.format(year_new,mon_new,day)
                else:
                    new_fname = south_modified_dir + '{}\\{:02d}{}.tif'.format(year,mon,day)

                # print old_fname
                # print new_fname
                try:
                    shutil.copy(old_fname,new_fname)
                except Exception as e:
                    print e
                    continue



    def split_files(self,fdir,outdir):
        Tools().mk_dir(outdir)
        years = np.array(range(1982, 2016))
        for y in tqdm(years):
            for f in os.listdir(fdir):
                year = int(f[:4])
                if y == year:
                    outdir_y = outdir + '{}\\'.format(y)
                    Tools().mk_dir(outdir_y)
                    shutil.copy(fdir + f, outdir_y + f)


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
                        array = np.array(array, dtype=float)
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
                void_dic['%03d.%03d' % (r, c)] = []
                void_dic_list.append('%03d.%03d' % (r, c))

        # print(len(void_dic))
        # exit()
        params = []
        for r in tqdm(range(row)):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic['%03d.%03d' % (r, c)].append(val) # TODO: need to be transformed into tuple

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

    def data_transform_split_files(self,fdir,outdir):
        for year in os.listdir(fdir):
            print year
            fdir_i = fdir + year + '\\'
            outdir_i = outdir + year + '\\'
            Tools().mk_dir(outdir_i, force=1)
            self.data_transform(fdir_i, outdir_i)

        pass

    def kernel_hants(self, params):
        outdir, y, fdir = params
        outdir_y = outdir + y + '\\'
        Tools().mk_dir(outdir_y, force=1)
        for f in os.listdir(fdir + y):
            dic = dict(np.load(fdir + y + '\\' + f).item())
            hants_dic = {}
            for pix in dic:
                vals = dic[pix]
                vals = np.array(vals)
                std = np.std(vals)
                if std == 0:
                    continue
                xnew, ynew = self.__interp__(vals)
                ynew = np.array([ynew])
                # print np.std(ynew)
                results = HANTS(sample_count=365, inputs=ynew, low=-10000, high=10000,
                                fit_error_tolerance=std)
                result = results[0]

                # plt.plot(result)
                # plt.plot(range(len(ynew[0])),ynew[0])
                # plt.show()
                hants_dic[pix] = result
            np.save(outdir_y + f, hants_dic)

    def hants(self):
        for hemi in ['north','south_modified']:
            outdir = self.this_class_arr + 'hants_smooth\\{}\\'.format(hemi)
            fdir = data_root + 'NDVI_phenology\\per_pix_separate\\{}\\'.format(hemi)
            params = []
            for y in os.listdir(fdir):
                params.append([outdir, y, fdir])
                # self.kernel_hants([outdir, y, fdir])
            MULTIPROCESS(self.kernel_hants, params).run(process=5)

    def check_hants(self):
        hemi = 'south_modified'
        fdir = self.this_class_arr + 'hants_smooth\\{}\\'.format(hemi)
        tropical_mask_dic = NDVI().tropical_mask_dic

        for year in os.listdir(fdir):
            perpix_dir = fdir + '{}\\'.format(year)
            for f in os.listdir(perpix_dir):
                if not '021' in f:
                    continue
                dic = T.load_npy(perpix_dir + f)
                for pix in dic:
                    if pix in tropical_mask_dic:
                        continue
                    vals = dic[pix]
                    if len(vals) > 0:
                        # print pix,vals
                        plt.plot(vals)
                        plt.show()
                        sleep()
            exit()

    def __interp__(self, vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = range(len(vals))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new, y_new

    def __search_left(self, vals, maxind, threshold_i):
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self, vals, maxind, threshold_i):
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def SOS_EOS(self, threshold_i=0.5):
        for hemi in ['north','south_modified']:
            out_dir = self.this_class_arr + 'SOS_EOS\\threshold_{}\\{}\\'.format(threshold_i,hemi)
            Tools().mk_dir(out_dir, force=1)
            # fdir = data_root + 'NDVI_phenology\\HANTS\\'
            fdir = self.this_class_arr + 'hants_smooth\\{}\\'.format(hemi)
            for y in tqdm(os.listdir(fdir)):
                year_dir = fdir + y + '\\'
                result_dic = {}
                for f in os.listdir(year_dir):
                    dic = dict(np.load(year_dir + f).item())
                    for pix in dic:
                        try:
                            vals = dic[pix]
                            maxind = np.argmax(vals)
                            start = self.__search_left(vals, maxind, threshold_i)
                            end = self.__search_right(vals, maxind, threshold_i)
                            result = [start,maxind, end]
                            result_dic[pix] = result
                            # print result
                            # sleep()
                        except:
                            pass
                            # plt.plot(vals)
                            # plt.show()
                            # exit()
                        # plt.plot(vals)
                        # plt.plot(range(start,end),vals[start:end],linewidth=4,zorder=99,color='r')
                        # plt.title('start:{} \nend:{} \nduration:{}'.format(start,end,end-start))
                        # plt.show()
                np.save(out_dir + y, result_dic)

    def check_SOS_EOS(self,threshold_i=0.5):
        fdir = self.this_class_arr + 'SOS_EOS\\threshold_{}\\'.format(threshold_i)
        for f in os.listdir(fdir):
            dic = T.load_npy(fdir+f)
            spatial_dic = {}
            for pix in dic:
                SOS = dic[pix][0]
                pix = (int(pix.split('.')[0]),int(pix.split('.')[1]))
                spatial_dic[pix] = SOS
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr)
            plt.show()
        pass


class Phenology_based_on_Temperature_NDVI:
    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Phenology_based_on_Temperature_NDVI\\'
        self.this_class_tif = results_root + 'tif\\Phenology_based_on_Temperature_NDVI\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        pass

    def run(self):
        # fdir = data_root + 'TMP\\mon_mean\\'
        # outdir = data_root + 'TMP\\annual_mean\\'
        # fdir = data_root + 'NDVI\\mon_mean\\'
        # outdir = data_root + 'NDVI\\annual_mean\\'
        # self.mon_tif_mean(fdir,outdir)
        # self.gen_phenology_valid_pixs()
        # self.cal_phenology_north_hemi()
        # self.cal_phenology_south_hemi()
        self.compose_south_north_hemi_phenology()

        pass


    def mon_tif_mean(self,fdir,outdir):
        # fdir = data_root + 'TMP\\mon_mean\\'
        # outdir = data_root + 'TMP\\annual_mean\\'
        T.mk_dir(outdir)
        outf = outdir + 'annual_mean'
        arrs = []
        for mon in range(1,13):
            f = fdir + '{:02d}.tif'.format(mon)
            array,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(f)
            # T.mask_999999_arr(array)
            arrs.append(array)
        void_dic = DIC_and_TIF().void_spatial_dic()
        for i in tqdm(range(len(arrs[0]))):
            for j in range(len(arrs[0][0])):
                vals = []
                for k in range(len(arrs)):
                    val = arrs[k][i][j]
                    # if np.isnan(val):
                    #     vals = []
                    #     break
                    vals.append(val)
                pix = (i,j)
                void_dic[pix] = vals
        np.save(outf,void_dic)
        pass


    def gen_phenology_valid_pixs(self):

        outdir = self.this_class_arr + 'valid_pixs\\'
        T.mk_dir(outdir)
        outf = outdir + 'valid_pixs'
        T_annual_mean_f = data_root + 'TMP\\annual_mean\\annual_mean.npy'
        NDVI_annual_mean_f = data_root + 'NDVI\\annual_mean\\annual_mean.npy'

        T_annual_mean_dic = T.load_npy(T_annual_mean_f)
        NDVI_annual_mean_dic = T.load_npy(NDVI_annual_mean_f)


        r_dic = {}
        for pix in tqdm(T_annual_mean_dic):
            T_annual_vals = T_annual_mean_dic[pix]
            NDVI_annual_vals = NDVI_annual_mean_dic[pix]
            if len(T_annual_vals) == 0 or len(NDVI_annual_vals) == 0:
                continue
            success = 1
            for val in T_annual_vals:
                if np.isnan(val):
                    success = 0
                    break
            if success == 0:
                continue

            new_T_annual_vals = []
            new_NDVI_annual_vals = []

            for i in range(len(NDVI_annual_vals)):
                if NDVI_annual_vals[i] < -999:
                    continue
                new_T_annual_vals.append(T_annual_vals[i])
                new_NDVI_annual_vals.append(NDVI_annual_vals[i])
            T_annual_vals = new_T_annual_vals
            NDVI_annual_vals = new_NDVI_annual_vals
            if len(T_annual_vals) < 10:
                continue
            r,p = stats.pearsonr(T_annual_vals,NDVI_annual_vals)
            if p > 0.01:
                continue
            if r < 0.5:
                continue
            r_dic[pix] = r
        np.save(outf,r_dic)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(r_dic)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(arr)
        # plt.show()
            # plt.plot(T_annual_vals,c='r',label='TMP')
            # plt.legend(loc=2)
            # plt.ylabel('TMP')
            # plt.twinx()
            # plt.plot(NDVI_annual_vals,c='green',label='NDVI')
            # plt.grid(1)
            # plt.ylabel('NDVI',rotation=-90)
            # plt.legend(loc=1)
            # plt.tight_layout()
            # plt.show()

        pass

    def check_consecutive(self,arr):
        consecutive = True
        for i in range(len(arr)):
            if i+1 == len(arr):
                break
            if arr[i+1] - arr[i] != 1:
                consecutive = False
        return consecutive


        pass


    def compose_south_north_hemi_phenology(self):
        outf = self.this_class_arr + 'growing_season_index'
        northf = self.this_class_arr + 'cal_phenology\\north.npy'
        southf = self.this_class_arr + 'cal_phenology\\south.npy'
        north_dic = T.load_npy(northf)
        south_dic = T.load_npy(southf)

        global_phe = {}
        for pix in north_dic:
            gs = north_dic[pix]
            global_phe[pix] = gs
        for pix in south_dic:
            gs = south_dic[pix]
            global_phe[pix] = gs

        np.save(outf,global_phe)

        spatial_dic = {}
        # gs_len_dic = {}
        for pix in global_phe:
            first_mon = global_phe[pix][0]
            length = len(global_phe[pix])
            spatial_dic[pix] = first_mon
            # gs_len_dic[pix] = length

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().arr_to_tif(arr,self.this_class_tif + 'first_mon_gs.tif')
        # len_arr = DIC_and_TIF().pix_dic_to_spatial_arr(gs_len_dic)
        # plt.figure()
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(arr)
        # plt.colorbar()
        #
        # plt.figure()
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(len_arr)
        # plt.colorbar()
        #
        # plt.show()

        pass

    def cal_phenology_north_hemi(self):
        outdir = self.this_class_arr + 'cal_phenology\\'
        outf = outdir + 'north'
        T.mk_dir(outdir)
        valid_pix_dic_f = self.this_class_arr + 'valid_pixs\\valid_pixs.npy'
        NDVI_annual_mean_f = data_root + 'NDVI\\annual_mean\\annual_mean.npy'
        pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
        lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
        NDVI_annual_mean_dic = T.load_npy(NDVI_annual_mean_f)
        valid_pix_dic = T.load_npy(valid_pix_dic_f)
        hemi_gs = {}
        for pix in valid_pix_dic:
            lon,lat = lon_lat_dic[pix]
            ndvi = NDVI_annual_mean_dic[pix]
            ndvi = np.array(ndvi)
            if lat < 0:
                continue
            # if lat < 0:
            #     a = ndvi[:6]
            #     b = ndvi[6:]
            #     ndvi = np.append(b,a)
            ndvi[ndvi<-9999] = -1000
            if np.max(ndvi) < 2000:
                continue

            ndvi_smooth = SMOOTH().smooth_convolve(ndvi,9)[1:]
            ndvi = ndvi_smooth
            median = np.median(ndvi)
            gs_index = []
            for i in range(len(ndvi)):
                if ndvi[i] > median:
                    gs_index.append(i+1)
            hemi_gs[pix] = gs_index
            # if not self.check_consecutive(gs_index):
            #     print lon,lat
            #     print gs_index
            #     print
            #     plt.plot(ndvi)
            #     plt.show()
        np.save(outf,hemi_gs)
            # ks = []
            # for i in range(len(ndvi)):
            #     if i + 1 == len(ndvi):
            #         continue
            #     k = ndvi[i+1] - ndvi[i]
            #     ks.append(k)
            # ks.insert(0,0)
            # ks_argsort = np.argsort(ks)
            # min_ks = ks[ks_argsort[0]]
            # max_ks = ks[ks_argsort[-1]]
            # plt.vlines(ks_argsort[0] + 1,min_ks,max_ks,linestyles='--', colors='r')
            # plt.vlines(ks_argsort[-1] + 1,min_ks,max_ks,linestyles='--', colors='r')
            # plt.plot(np.array(range(len(ks)))+1,ks,'--o',c='gray')
            # plt.twinx()
            # median = np.median(ndvi)
            # summer_start = max(ndvi) * 0.95
            # summer_end = max(ndvi) * 0.85
            # plt.hlines(summer_start, 1, len(ndvi), linestyles='--', colors='green')
            # plt.hlines(summer_end, 1, len(ndvi), linestyles='--', colors='b')
            # plt.hlines(median, 1, len(ndvi))
            # plt.plot(np.array(range(len(ndvi)))+1,ndvi,'-o',c='g')
            # plt.grid(1)
            # plt.show()
        pass


    def cal_phenology_south_hemi(self):
        outdir = self.this_class_arr + 'cal_phenology\\'
        outf = outdir + 'south'
        T.mk_dir(outdir)
        valid_pix_dic_f = self.this_class_arr + 'valid_pixs\\valid_pixs.npy'
        NDVI_annual_mean_f = data_root + 'NDVI\\annual_mean\\annual_mean.npy'
        pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
        lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
        NDVI_annual_mean_dic = T.load_npy(NDVI_annual_mean_f)
        valid_pix_dic = T.load_npy(valid_pix_dic_f)
        hemi_gs = {}
        for pix in valid_pix_dic:
            lon,lat = lon_lat_dic[pix]
            ndvi = NDVI_annual_mean_dic[pix]
            ndvi = np.array(ndvi)
            if lat >= 0:
                continue
            a = ndvi[:6]
            b = ndvi[6:]
            ndvi = np.append(b,a)
            ndvi[ndvi<-9999] = -1000
            if np.max(ndvi) < 2000:
                continue

            ndvi_smooth = SMOOTH().smooth_convolve(ndvi,9)[1:]
            ndvi = ndvi_smooth
            median = np.median(ndvi)
            gs_index = []
            for i in range(len(ndvi)):
                if ndvi[i] > median:
                    gs_index.append(i+1)
            new_gs = []
            for i in gs_index:
                gs = i + 6
                if gs > 12:
                    gs = gs - 12
                new_gs.append(gs)
            hemi_gs[pix] = new_gs
        np.save(outf,hemi_gs)





def main():
    # Phenology().run()
    Phenology_based_on_Temperature_NDVI().run()
    pass


if __name__ == '__main__':
    main()
