# coding=gbk

from __init__ import *


def hdf_to_tif():
    out_dir = this_root+'data\\NDVI\\tif_8km_bi_weekly\\'
    Tools().mk_dir(out_dir)
    fdir = this_root+'data\\NDVI\\HDF\\'
    for f in tqdm(os.listdir(fdir)):
        if not f.endswith('.hdf'):
            continue
        # if not f == 'ndvi3g_geo_v1_2014_0106.hdf':
        #     continue
        # print(f)
        year = f.split('.')[0].split('_')[-2]
        # print(year)
        hdf = h5py.File(fdir+f, 'r')
        for i in range(len(hdf['time'])):
            arr = hdf['ndvi'][i]
            lon = hdf['lon']
            lat = hdf['lat']
            time = hdf['time'][i]
            time_str = str(time)
            if time_str.endswith('.5'):
                date = year+'%02d'%int(time)+'15'
            else:
                date = year+'%02d'%int(time)+'01'

            newRasterfn = out_dir+'{}.tif'.format(date)
            longitude_start = lon[0]
            latitude_start = lat[0]
            pixelWidth = lon[1]-lon[0]
            pixelHeight = lat[1]-lat[0]
            arr = np.array(arr,dtype=float)
            # print(arr.dtype)
            grid = arr > - 10000
            arr[np.logical_not(grid)] = -999999
            # import time
            # time.sleep(1)
            # plt.imshow(arr)
            # plt.show()
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)
            pass



def nc_dir_to_tif(ncdir,outdir,variable):
    Tools().mk_dir(outdir)
    for nc in os.listdir(ncdir):
        outdir_i = outdir+nc.split('.')[0]+'\\'
        Tools().mk_dir(outdir_i)
        ncin = Dataset(ncdir+nc, 'r')
        lat = ncin['lat'][::-1]
        lon = ncin['lon']
        pixelWidth = lon[1]-lon[0]
        pixelHeight = lat[1]-lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]

        time = ncin.variables['time']

        # print(time)
        # exit()
        # time_bounds = ncin.variables['time_bounds']
        # print(time_bounds)
        start = datetime.datetime(1900, 01, 01)
        # a = start + datetime.timedelta(days=5459)
        # print(a)
        # print(len(time_bounds))
        # print(len(time))
        # for i in time:
        #     print(i)
        # exit()
        # nc_dic = {}
        flag = 0

        valid_year = []
        for i in range(1982, 2016):
            valid_year.append(str(i))

        for i in tqdm(range(len(time)),desc=nc.split('.')[0]):
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time[i]))
            year = str(date.year)
            month = '%02d' % date.month
            # day = '%02d'%date.day
            date_str = year + month
            if not date_str[:4] in valid_year:
                continue
            # print(date_str)
            arr = ncin.variables[variable][i][::-1]
            arr = np.array(arr)
            grid = arr < 99999
            arr[np.logical_not(grid)] = -999999
            newRasterfn = outdir_i+date_str+'.tif'
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # plt.imshow(arr,'RdBu',vmin=-3,vmax=3)
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()


def nc_f_dir_to_tif(ncf,outdir,variable):
    nc = ncf
    ncin = Dataset(nc, 'r')
    lat = ncin['lat'][::-1]
    lon = ncin['lon']
    pixelWidth = lon[1]-lon[0]
    pixelHeight = lat[1]-lat[0]
    longitude_start = lon[0]
    latitude_start = lat[0]

    time = ncin.variables['time']

    # print(time)
    # exit()
    # time_bounds = ncin.variables['time_bounds']
    # print(time_bounds)
    start = datetime.datetime(1900, 01, 01)
    # a = start + datetime.timedelta(days=5459)
    # print(a)
    # print(len(time_bounds))
    # print(len(time))
    # for i in time:
    #     print(i)
    # exit()
    # nc_dic = {}
    flag = 0


    for i in tqdm(range(len(time)),desc=nc.split('.')[0]):
        flag += 1
        # print(time[i])
        date = start + datetime.timedelta(days=int(time[i]))
        year = str(date.year)
        month = '%02d' % date.month
        # day = '%02d'%date.day
        date_str = year + month
        # if not date_str[:4] in valid_year:
        #     continue
        # print(date_str)
        arr = ncin.variables[variable][i][::-1]
        arr = np.array(arr)
        grid = arr < 99999
        arr[np.logical_not(grid)] = -999999
        newRasterfn = outdir+date_str+'.tif'
        to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,arr)
        # grid = np.ma.masked_where(grid>1000,grid)
        # plt.imshow(arr,'RdBu',vmin=-3,vmax=3)
        # plt.colorbar()
        # plt.show()
        # nc_dic[date_str] = arr
        # exit()



def kernel_bi_weekly_to_monthly(params):

    y,fdir,outdir = params
    for m in range(1, 13):
        date = '{}{:02d}'.format(y, m)
        one_month_tif = []
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            date_i = f[:6]
            if date_i == date:
                one_month_tif.append(f)
        arrs = []
        for tif in one_month_tif:
            arr = to_raster.raster2array(fdir + tif)[0]
            arr[arr < -2999] = np.nan
            arrs.append(arr)

        one_month_mean = []
        for i in range(len(arrs[0])):
            temp = []
            for j in range(len(arrs[0][0])):
                sum_ = []
                for k in range(len(arrs)):
                    val = arrs[k][i][j]
                    sum_.append(val)
                mean = np.nanmean(sum_)
                temp.append(mean)
            one_month_mean.append(temp)
        one_month_mean = np.array(one_month_mean)
        DIC_and_TIF().arr_to_tif(one_month_mean, outdir + '{}.tif'.format(date))


def bi_weekly_to_monthly():
    fdir = this_root+'data\\NDVI\\tif_05deg_bi_weekly\\'
    outdir = this_root+'data\\NDVI\\tif\\'
    Tools().mk_dir(outdir)

    params = []
    for y in range(1982,2016):
        params.append([y,fdir,outdir])
    MULTIPROCESS(kernel_bi_weekly_to_monthly,params).run(desc='bi_weekly_to_monthly')


    pass




def data_transform(fdir, outdir):
    # 不可并行，内存不足
    Tools().mk_dir(outdir,force=1)
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




def data_transform_swe(fdir, outdir):
    # 不可并行，内存不足
    Tools().mk_dir(outdir,force=1)
    # 将空间图转换为数组
    # per_pix_data
    date_list = []
    for y in range(1982,2016):
        for m in range(1,13):
            date = '{}{:02d}'.format(y,m)
            date_list.append(date)
    all_array = []
    for d in tqdm(date_list, 'loading...'):
        # for d in date_list:
        tif = fdir+d+'.tif'
        if not os.path.isfile(tif):
            void_dic = D.void_spatial_dic_nan()
            arr = D.pix_dic_to_spatial_arr(void_dic)
            all_array.append(arr)
            continue
        # print(d)
        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)
        array = np.array(array,dtype=np.float)
        array[array<=0]=np.nan
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



def check_data_transform():
    fdir = data_root+'SPEI\\per_pix\\spei03\\'
    for f in os.listdir(fdir):
        dic = T.load_npy(fdir+f)

        for pix in dic:
            vals = dic[pix]
            if vals[0]<=-9999:
                continue

            vals = np.array(vals,dtype=np.float)
            vals[vals<=-9999]=np.nan
            plt.plot(vals)
            plt.show()




class CleanData:

    def __init__(self):

        pass

    def run(self):
        # x = ['NDVI','PRE','TMP','SM']
        x = ['VPD']
        for i in x:
            print i
            self.clean_origin_vals(i)
        # self.clean_origin_vals_SWE('SWE')
        # self.check_clean()
        # self.clean_SPEI()
        pass


    def clean_origin_vals(self,x):
        fdir = data_root+'{}\\per_pix\\'.format(x)
        outdir = data_root+'{}\\per_pix_clean\\'.format(x)
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir+f)
            clean_dic = {}
            for pix in dic:
                val = dic[pix]
                val = np.array(val,dtype=np.float)
                val[val<-9999]=np.nan
                new_val = T.interp_nan(val,kind='linear')
                if len(new_val) == 1:
                    continue
                # plt.plot(val)
                # plt.show()
                clean_dic[pix] = new_val
            np.save(outdir+f,clean_dic)

    def clean_origin_vals_SWE(self,x='SWE'):
        fdir = data_root+'{}\\per_pix\\'.format(x)
        outdir = data_root+'{}\\per_pix_clean\\'.format(x)
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir+f)
            clean_dic = {}
            for pix in dic:
                val = dic[pix]
                val = np.array(val,dtype=np.float)
                val_filter = T.interp_nan(val)
                if len(val_filter) == 1:
                    continue
                new_val = []
                for i in val:
                    if np.isnan(i):
                        v = 0
                    else:
                        v = i
                    new_val.append(v)
                # plt.plot(new_val)
                # plt.show()
                clean_dic[pix] = new_val
            np.save(outdir+f,clean_dic)


    def check_clean(self):
        x = 'SWE'
        fdir = data_root + '{}\\per_pix_clean\\'.format(x)
        x_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir+f)
            for key in dic:
                if len(dic[key]) == 0:
                    continue
                x_dic[key] = np.mean(dic[key])
        arr = D.pix_dic_to_spatial_arr(x_dic)
        # plt.imshow(arr,vmin=0,vmax=100) #pre
        # plt.imshow(arr,vmin=-30,vmax=30) # tmp
        # plt.imshow(arr,vmin=0,vmax=0.3) # sm
        plt.imshow(arr)
        plt.colorbar()
        plt.show()

    def clean_SPEI(self):
        for i in range(1,13):
            fdir = data_root + 'SPEI\\per_pix\\spei{:02d}\\'.format(i)
            outdir = data_root + 'SPEI\\per_pix_clean\\spei{:02d}\\'.format(i)
            T.mk_dir(outdir,force=1)
            for f in tqdm(os.listdir(fdir),desc=str(i)):
                dic = T.load_npy(fdir + f)
                clean_dic = {}
                for pix in dic:
                    val = dic[pix]
                    val = np.array(val, dtype=np.float)
                    val[val < -9999] = np.nan
                    # new_val = T.interp_nan(val, kind='linear')
                    new_val = T.interp_nan(val)
                    if len(new_val) == 1:
                        continue
                    # plt.plot(val)
                    # plt.show()
                    clean_dic[pix] = new_val
                np.save(outdir + f, clean_dic)


        pass

def smooth_x():
    for x in ['NDVI','PRE','TMP','SM']:
        fdir = data_root+'{}\\per_pix_clean_anomaly\\'.format(x)
        outdir = data_root+'{}\\per_pix_clean_anomaly_smooth\\'.format(x)
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir),desc='{}'.format(x)):
            dic = T.load_npy(fdir+f)
            smooth_dic = {}
            for pix in dic:
                vals = dic[pix]
                smooth_val = S.forward_window_smooth(vals)
                smooth_dic[pix] = smooth_val
            np.save(outdir+f,smooth_dic)

def cal_anomaly_x():
    # for x in ['NDVI','PRE','TMP','SM']:
    for x in ['SWE','VPD']:
        print x
        fdir = data_root+'{}\\per_pix_clean\\'.format(x)
        outdir = data_root+'{}\\per_pix_clean_anomaly\\'.format(x)
        Pre_Process().cal_anomaly(fdir,outdir)



def check_anomaly_x():
    for x in ['NDVI','PRE','TMP','SM']:
        print x
        # x = 'PRE'
        fdir = data_root+'{}\\per_pix_clean_anomaly\\'.format(x)
        x_dic = {}
        for f in os.listdir(fdir):
            if not '015' in f:
                continue
            dic = T.load_npy(fdir+f)
            for pix in dic:
                print pix,
                val = dic[pix]
                plt.plot(val)
                plt.show()


def smooth_SPEI():
    for i in range(1,13):
        fdir = data_root+'SPEI\\per_pix_clean\\spei{:02d}\\'.format(i)
        outdir = data_root+'SPEI\\per_pix_clean_smooth\\spei{:02d}\\'.format(i)
        T.mk_dir(outdir,force=1)
        for f in tqdm(os.listdir(fdir),desc='{}'.format(i)):
            dic = T.load_npy(fdir+f)
            smooth_dic = {}
            for pix in dic:
                vals = dic[pix]
                smooth_val = S.forward_window_smooth(vals)
                smooth_dic[pix] = smooth_val
            np.save(outdir+f,smooth_dic)


def split_files():

    fdir = this_root+'data\\NDVI\\tif_0.25_bi_weekly\\'
    outdir = this_root+'data\\NDVI\\tif_0.25_bi_weekly_yearly\\'
    Tools().mk_dir(outdir)
    years = np.array(range(1982,2016))
    for y in tqdm(years):
        for f in os.listdir(fdir):
            year = int(f[:4])
            if y == year:
                outdir_y = outdir+'{}\\'.format(y)
                Tools().mk_dir(outdir_y)
                shutil.copy(fdir+f,outdir_y+f)


def data_transform_split_files():
    fdir = this_root+'data\\NDVI\\tif_0.25_bi_weekly_yearly\\'
    outdir = this_root+'data\\NDVI\\per_pix_0.25_bi_weekly_yearly\\'
    for year in os.listdir(fdir):
        print year
        fdir_i = fdir+year+'\\'
        outdir_i = outdir+year+'\\'
        Tools().mk_dir(outdir_i,force=1)
        data_transform(fdir_i,outdir_i)

    pass


def swe_unify():
    '''
    将swe无效值改为-999999
    :return:
    '''
    fdir = data_root+'{}\\tif_origin\\'.format('SWE')
    outdir = data_root+'{}\\tif\\'.format('SWE')
    T.mk_dir(outdir)
    date_list = []
    for y in range(1982,2016):
        for m in range(1,6)+range(10,13):
            date = '{}{:02d}'.format(y,m)
            date_list.append(date)
    for f in date_list:
        f = f+'.tif'
        print f

        arr = to_raster.raster2array(fdir+f)[0]
        arr[arr<=0]=np.nan
        D.arr_to_tif(arr,outdir+f)
    pass

def gen_monthly_mean(x):
    fdir = data_root+'{}\\tif\\'.format(x)
    outdir = data_root+'{}\\mon_mean\\'.format(x)
    T.mk_dir(outdir)
    for mon in tqdm(range(1, 13)):
        # print x,mon
        arr_sum = []
        flag = 0
        void_dic = D.void_spatial_dic()
        for year in range(1982,2016):
            f = fdir+'{}{:02d}.tif'.format(year,mon)
            if not os.path.isfile(f):
                continue
            flag += 1
            arr = to_raster.raster2array(f)[0]
            T.mask_999999_arr(arr)
            dic = D.spatial_arr_to_dic(arr)
            for pix in dic:
                void_dic[pix].append(dic[pix])
        mon_mean_dic = {}
        for pix in void_dic:
            vals = void_dic[pix]
            mean_vals = np.nanmean(vals)
            mon_mean_dic[pix] = mean_vals
        mon_mean = D.pix_dic_to_spatial_arr(mon_mean_dic)
        D.arr_to_tif(mon_mean,outdir+'{:02d}.tif'.format(mon))
    pass


def NDVI_detrend():
    fdir = data_root+'NDVI\\per_pix_clean_anomaly_smooth\\'
    outdir =data_root+'NDVI\\per_pix_clean_anomaly_smooth_detrend\\'
    T.mk_dir(outdir)
    for f in tqdm(os.listdir(fdir)):
        dic = T.load_npy(fdir+f)
        detrend_dic = T.detrend_dic(dic)
        np.save(outdir+f,detrend_dic)
    pass


def SPEI_detrend():
    fdir = data_root + 'SPEI\\per_pix_clean_smooth\\'
    outdir = data_root + 'SPEI\\per_pix_clean_smooth_detrend\\'
    T.mk_dir(outdir)
    for interval in tqdm(os.listdir(fdir)):
        spei_dir = fdir+'{}\\'.format(interval)
        outdir_i = outdir+'{}\\'.format(interval)
        T.mk_dir(outdir_i)
        for f in os.listdir(spei_dir):
            dic = T.load_npy(spei_dir+f)
            detrend_dic = T.detrend_dic(dic)
            np.save(outdir_i+f,detrend_dic)
    pass


class CWD:

    def __init__(self):

        pass

    def run(self):
        # self.nc_to_tif()
        # self.per_pix_trans()
        # self.per_pix_clean()
        # self.p_minus_pet()
        # self.check_p_minus_pet()
        # self.cal_anomaly()
        # self.smooth_cwd()
        # self.detrend_cwd()
        # self.pick_1982_2015()
        self.pick_1982_2015_detrend()
        # self.check_cwd()
        pass

    def nc_to_tif(self):
        pre_nc_f = data_root+'CWD\\nc\\cru_ts4.04.1901.2019.pre.dat.nc'
        pet_nc_f = data_root+'CWD\\nc\\cru_ts4.04.1901.2019.pet.dat.nc'

        pre_out_dir = data_root+'CWD\\tif\\pre\\'
        pet_out_dir = data_root+'CWD\\tif\\pet\\'
        T.mk_dir(pre_out_dir,force=True)
        T.mk_dir(pet_out_dir,force=True)
        nc_f_dir_to_tif(pre_nc_f,pre_out_dir,'pre')
        nc_f_dir_to_tif(pet_nc_f,pet_out_dir,'pet')

    def per_pix_trans(self):
        product = ['pre','pet']
        for p in product:
            fdir = data_root+'CWD\\tif\\{}\\'.format(p)
            outdir = data_root+'CWD\\per_pix_data\\{}\\'.format(p)
            data_transform(fdir,outdir)
            pass

    def per_pix_clean(self):
        # fdir = data_root+'{}\\per_pix\\'.format(x)
        product = ['pre', 'pet']
        for p in product:
            fdir = data_root+'CWD\\per_pix_data\\{}\\'.format(p)
            outdir = data_root+'CWD\\per_pix_data_clean\\{}\\'.format(p)
            T.mk_dir(outdir,force=True)
            for f in tqdm(os.listdir(fdir)):
                dic = T.load_npy(fdir+f)
                clean_dic = {}
                for pix in dic:
                    val = dic[pix]
                    val = np.array(val,dtype=np.float)
                    val[val<-9999]=np.nan
                    new_val = T.interp_nan(val,kind='linear')
                    if len(new_val) == 1:
                        continue
                    # plt.plot(val)
                    # plt.show()
                    clean_dic[pix] = new_val
                np.save(outdir+f,clean_dic)

    def p_minus_pet(self):
        pre_dir = data_root+'CWD\\per_pix_data_clean\\pre\\'
        pet_dir = data_root+'CWD\\per_pix_data_clean\\pet\\'
        outdir = data_root+'CWD\\per_pix_p_minus_pet\\'
        T.mk_dir(outdir)
        dic_pre = {}
        dic_pet = {}
        for f in tqdm(os.listdir(pre_dir),desc='loading data'):
            dic_pre_i = T.load_npy(pre_dir+f)
            dic_pet_i = T.load_npy(pet_dir+f)
            dic_pre.update(dic_pre_i)
            dic_pet.update(dic_pet_i)

        p_minus_pet_dic = {}
        for pix in tqdm(dic_pre,desc='calculating p - pet'):
            if not pix in dic_pet:
                continue
            pre = dic_pre[pix]
            pet = dic_pet[pix]
            pre = np.array(pre)
            pet = np.array(pet)
            # sleep()
            cwd = pre - pet
            p_minus_pet_dic[pix] = cwd
        np.save(outdir + 'p_minus_pet',p_minus_pet_dic)

    def check_p_minus_pet(self):
        f = data_root+'CWD\\per_pix_p_minus_pet\\p_minus_pet.npy'
        dic = T.load_npy(f)
        for pix in dic:
            print pix,dic[pix]
            plt.plot(dic[pix])
            plt.grid(1)
            plt.show()
            sleep()
        pass

    def cal_anomaly(self):
        f = data_root + 'CWD\\per_pix_p_minus_pet\\p_minus_pet.npy'
        outdir = data_root + 'CWD\\per_pix\\'
        Tools().mk_dir(outdir)
        pix_dic = dict(np.load(f).item())
        anomaly_pix_dic = {}
        for pix in tqdm(pix_dic):
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

            # 算法2
            pix_anomaly = []
            for i in range(len(vals)):
                mon = i % 12
                std_ = climatology_std[mon]
                mean_ = climatology_means[mon]
                if std_ == 0:
                    anomaly = 0  ##### 修改gpp
                else:
                    anomaly = (vals[i] - mean_) / std_

                pix_anomaly.append(anomaly)
            # pix_anomaly = Tools().interp_1d_1(pix_anomaly,-100)
            # plt.plot(pix_anomaly)
            # plt.show()
            pix_anomaly = np.array(pix_anomaly)
            anomaly_pix_dic[pix] = pix_anomaly

        np.save(outdir + 'CWD', anomaly_pix_dic)


        pass


    def smooth_cwd(self):

        f = data_root + 'CWD\\per_pix\\CWD.npy'
        outdir = data_root + 'CWD\\per_pix_smooth\\'
        T.mk_dir(outdir)
        outf = outdir + 'CWD.npy'
        dic = T.load_npy(f)
        smooth_dic = {}
        for pix in tqdm(dic):
            vals = dic[pix]
            smooth_vals = SMOOTH().forward_window_smooth(vals)
            smooth_dic[pix] = smooth_vals
        np.save(outf,smooth_dic)
        pass

    def detrend_cwd(self):
        dir = data_root + 'CWD\\per_pix_smooth\\'
        outdir = data_root + 'CWD\\per_pix_smooth_detrend\\'
        outf = outdir+'CWD.npy'
        T.mk_dir(outdir)
        f = dir + 'CWD.npy'
        dic = T.load_npy(f)
        detrend_dic = T.detrend_dic(dic)
        np.save(outf,detrend_dic)
        pass


    def pick_1982_2015(self):
        outdir = data_root+'CWD\\per_pix_1982_2015\\'
        outf = outdir+'CWD.npy'
        T.mk_dir(outdir)
        f = data_root+'CWD\\per_pix_smooth\\CWD.npy'
        dic = T.load_npy(f)
        base_time = datetime.datetime(1901,1,15)
        end_time = datetime.datetime(2019,12,15)
        date_range = end_time - base_time
        date_range_days = date_range.days
        dates = []
        for i in range(date_range_days):
            time_delta = datetime.timedelta(i)
            current_date = base_time + time_delta
            year = current_date.year
            month = current_date.month
            dates.append((year,month))
        dates = list(set(dates))
        dates.sort()
        selected_dates = []
        for y in range(1982,2016):
            for m in range(1,13):
                date = (y,m)
                selected_dates.append(date)

        selected_index = []
        for i,date_i in enumerate(dates):
            for date_j in selected_dates:
                if date_i == date_j:
                    selected_index.append(i)
        selected_dic = {}
        for pix in dic:
            vals = dic[pix]
            selected_vals = T.pick_vals_from_1darray(vals,selected_index)
            selected_vals = np.array(selected_vals)
            selected_dic[pix] = selected_vals

        np.save(outf,selected_dic)
        pass


    def pick_1982_2015_detrend(self):
        outdir = data_root+'CWD\\per_pix_1982_2015_detrend\\'
        outf = outdir+'CWD.npy'
        T.mk_dir(outdir)
        f = data_root+'CWD\\per_pix_smooth_detrend\\CWD.npy'
        dic = T.load_npy(f)
        base_time = datetime.datetime(1901,1,15)
        end_time = datetime.datetime(2019,12,15)
        date_range = end_time - base_time
        date_range_days = date_range.days
        dates = []
        for i in range(date_range_days):
            time_delta = datetime.timedelta(i)
            current_date = base_time + time_delta
            year = current_date.year
            month = current_date.month
            dates.append((year,month))
        dates = list(set(dates))
        dates.sort()
        selected_dates = []
        for y in range(1982,2016):
            for m in range(1,13):
                date = (y,m)
                selected_dates.append(date)

        selected_index = []
        for i,date_i in enumerate(dates):
            for date_j in selected_dates:
                if date_i == date_j:
                    selected_index.append(i)
        selected_dic = {}
        for pix in dic:
            vals = dic[pix]
            selected_vals = T.pick_vals_from_1darray(vals,selected_index)
            selected_vals = np.array(selected_vals)
            selected_dic[pix] = selected_vals

        np.save(outf,selected_dic)
        pass






    def check_cwd(self):
        start = time.time()
        f = data_root+'CWD\\per_pix_smooth\\CWD.npy'
        pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
        pix_to_lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
        dic = T.load_npy(f)
        end = time.time()
        print '{:0.2f}'.format(end-start)
        spatial_dic = {}
        for key in dic:
            print key
            lon,lat = pix_to_lon_lat_dic[key]

            val = dic[key]
            spatial_dic[key] = 1
            plt.plot(val)
            plt.grid(1)
            plt.title('lon {} lat {}'.format(lon,lat))
            plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()
        pass



def main():
    # hdf_to_tif()
    # bi_weekly_to_monthly()
    # split_files()
    # data_transform_split_files()
    # for x in ['PRE','TMP','SM']:
    # for x in ['VPD']:
    #     fdir = this_root + 'data\\{}\\tif\\'.format(x)
    #     outdir = this_root + 'data\\{}\\per_pix\\'.format(x)
    #     data_transform(fdir, outdir)
    # for i in range(1,13):
    #     fdir = this_root+'data\\SPEI\\tif\\spei{:02d}\\'.format(i)
    #     outdir = this_root+'data\\SPEI\\per_pix\\spei{:02d}\\'.format(i)
    #     data_transform(fdir,outdir)
    # x = 'SWE'
    # fdir = this_root + 'data\\{}\\tif\\'.format(x)
    # outdir = this_root + 'data\\{}\\per_pix\\'.format(x)
    # data_transform_swe(fdir,outdir)
    # check_data_transform()
    # nc_dir = this_root+'data\\SPEI\\download_from_web\\'
    # outdir = this_root+'data\\SPEI\\spei_tif\\'
    # variable = 'spei'
    # nc_to_tif(ncdir=nc_dir,outdir=outdir,variable=variable)
    # smooth_x()
    # cal_anomaly_x()
    # check_anomaly_x()
    # CleanData().run()
    # smooth_SPEI()
    # swe_unify()
    # param = []
    # for x in ['PRE', 'TMP', 'SM', 'SWE']:
    # for x in ['VPD']:
    #     # print x
    #     param.append(x)
    # M(gen_monthly_mean,param).run()
    #     gen_monthly_mean(x)
    # NDVI_detrend()
    # SPEI_detrend()
    CWD().run()
    pass



if __name__ == '__main__':
    main()