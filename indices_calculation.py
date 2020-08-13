# coding=gbk
from __init__ import *


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
        self.cal_gs()
        self.plot_gs()


        pass

    def plot_gs(self):
        f = self.this_class_arr + 'cal_gs\\gs.npy'
        dic = T.load_npy(f)
        spatial_dic = DIC_and_TIF().void_spatial_dic_nan()
        for pix in dic:
            first = dic[pix][0]
            spatial_dic[pix] = first
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
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
            # plt.plot(T_annual_vals, c='r', label='T')
            # plt.twinx()
            # plt.plot(NDVI_annual_vals, c='g', label='NDVI')
            # plt.title(str(r))
            # plt.show()
        np.save(outf,r_dic)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(r_dic)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(arr)
        # plt.colorbar()
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

    def cal_gs(self):
        outdir = self.this_class_arr + 'cal_gs\\'
        outf = outdir + 'gs'
        T.mk_dir(outdir)
        valid_pix_dic_f = self.this_class_arr + 'valid_pixs\\valid_pixs.npy'
        NDVI_annual_mean_f = data_root + 'NDVI\\annual_mean\\annual_mean.npy'
        # pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
        # lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
        NDVI_annual_mean_dic = T.load_npy(NDVI_annual_mean_f)
        valid_pix_dic = T.load_npy(valid_pix_dic_f)
        hemi_gs = {}
        for pix in tqdm(valid_pix_dic,desc='cal gs...'):
            # lon,lat = lon_lat_dic[pix]
            ndvi = NDVI_annual_mean_dic[pix]
            ndvi = np.array(ndvi)
            # if lat < 0:
            #     continue
            # if lat < 0:
            #     a = ndvi[:6]
            #     b = ndvi[6:]
            #     ndvi = np.append(b,a)
            ndvi[ndvi<-9999] = -1000
            if np.max(ndvi) < 2000:
                continue

            # ndvi_smooth = SMOOTH().smooth_convolve(ndvi,9)[1:]
            # ndvi = ndvi_smooth
            median = np.median(ndvi)
            if median < 2000:
                continue

            gs_index = []
            for i in range(len(ndvi)):
                if ndvi[i] > median:
                    gs_index.append(i+1)
            # print gs_index
            # plt.plot(ndvi)
            # plt.scatter(range(len(ndvi)), ndvi)
            # plt.hlines(median, 0, len(ndvi))
            # plt.show()
            if not self.check_consecutive(gs_index):
                continue
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


class Pick_drought_events:
    '''
    分月份
    '''
    def __init__(self):
        self.this_class_arr = results_root + 'arr\\Pick_drought_events\\'
        self.this_class_tif = results_root + 'tif\\Pick_drought_events\\'
        self.this_class_png = results_root + 'png\\Pick_drought_events\\'
        Tools().mk_dir(self.this_class_arr, force=True)
        Tools().mk_dir(self.this_class_tif, force=True)
        Tools().mk_dir(self.this_class_png, force=True)
        pass


    def run(self):

        #############################################################
        # ********************pick cwd events************************
        #############################################################
        # drought_index_f = data_root + 'CWD\\per_pix_1982_2015_detrend\\CWD.npy'
        # outdir = self.this_class_arr + 'cwd_events\\'
        # self.pick(drought_index_f, outdir)
        #############################################################
        # ************compose_spei_n_to_one_file*********************
        #############################################################
        # self.compose_spei_n_to_one_file()

        # spei_compose_dir = data_root + 'SPEI\\compose_spei_n_to_one_file\\'
        # outdir = self.this_class_arr + 'SPEI_events\\'
        # T.mk_dir(outdir)
        # for f in os.listdir(spei_compose_dir):
        #     drought_index_f = spei_compose_dir+f
        #     outdir_i = outdir + f.split('.')[0] + '\\'
        #     self.pick(drought_index_f,outdir_i)

        #############################################################
        # ***********************compose CWD SPEI********************
        #############################################################
        # self.check_events()
        self.compose_cwd_spei()

        pass

    def compose_cwd_spei(self):
        outdir = self.this_class_arr + 'compose_cwd_spei\\'
        outf = outdir + 'compose_cwd_spei'
        T.mk_dir(outdir)
        cwd_f = self.this_class_arr + 'cwd_events\\single_events.npy'
        cwd_dic_i = T.load_npy(cwd_f)

        spei_dir = self.this_class_arr + 'SPEI_events\\'
        compose_dic = DIC_and_TIF().void_spatial_dic()
        for folder in tqdm(os.listdir(spei_dir),desc='appending spei...'):
            f = os.path.join(spei_dir,folder,'single_events.npy')
            dic = T.load_npy(f)
            for pix in dic:
                events = dic[pix]
                for event in events:
                    compose_dic[pix].append(tuple(event))

        for pix in cwd_dic_i:
            events = cwd_dic_i[pix]
            for event in events:
                compose_dic[pix].append(tuple(event))

        no_duplicate_compose_dic = {}
        for pix in tqdm(compose_dic):
            events = compose_dic[pix]
            if len(events) == 0:
                continue
            events = set(events)
            df = pd.DataFrame()
            df_event = []
            df_event_mean = []
            for event in events:
                mean = np.mean(event)
                df_event.append(event)
                df_event_mean.append(mean)
            df['mean'] = df_event_mean
            df['event'] = df_event
            df.sort_values(by='mean',inplace=True)

            means = df['mean']
            events = df['event']
            means = list(means)
            events = list(events)
            means.sort()
            cluster = []
            clusters = []
            for i in range(len(means)):
                if i + 1 == len(means):
                    if len(cluster)>0:
                        clusters.append(cluster)
                    a = means[i-1]
                    b = means[i]
                    if abs(a - b) > 12:
                        clusters.append([i])
                    continue
                a = means[i]
                b = means[i + 1]
                if abs(a-b) < 12:
                    cluster.append(i)
                    cluster.append(i+1)
                else:
                    if len(cluster) == 0:
                        clusters.append([i])
                        continue
                    cluster = list(set(cluster))
                    cluster.sort()
                    clusters.append(cluster)
                    cluster = []
            drought_ranges = []
            for indxs in clusters:
                selected_indx = indxs[0]
                drought_range = events[selected_indx]
                drought_ranges.append(drought_range)
            no_duplicate_compose_dic[pix] = drought_ranges
        print 'saving dic...'
        np.save(outf,no_duplicate_compose_dic)
    def check_events(self):
        fdir = self.this_class_arr + 'cwd_events\\'
        for f in os.listdir(fdir):
            dic = T.load_npy(fdir + f)
            for pix in dic:
                print pix,dic[pix]
                sleep()



    def compose_spei_n_to_one_file(self):
        fdir = data_root + 'SPEI\\per_pix_clean_smooth\\'
        outdir = data_root + 'SPEI\\compose_spei_n_to_one_file\\'
        T.mk_dir(outdir)
        for spei in tqdm(os.listdir(fdir)):
            dic = {}
            for f in os.listdir(fdir + spei):
                dic_i = T.load_npy(os.path.join(fdir,spei,f))
                dic.update(dic_i)
            np.save(outdir + spei,dic)
        pass


    def check_EGS_LGS_events(self):
        event_dic_f = self.this_class_arr + 'EGS_LGS_events\\EGS_LGS_events.npy'
        event_dic = T.load_npy(event_dic_f)

        pre_dic = {}
        early_dic = {}
        late_dic = {}
        for pix in event_dic:
            pre_num = 0
            early_num = 0
            late_num = 0
            for eln,drought_range in event_dic[pix]:
                if eln == 'pre':
                    pre_num += 1
                elif eln == 'early':
                    early_num += 1
                elif eln == 'late':
                    late_num += 1
                else:
                    pass
            if pre_num > 0:
                pre_dic[pix] = pre_num
            if early_num > 0:
                early_dic[pix] = early_num
            if late_num > 0:
                late_dic[pix] = late_num
        pre_arr = DIC_and_TIF().pix_dic_to_spatial_arr(pre_dic)
        early_arr = DIC_and_TIF().pix_dic_to_spatial_arr(early_dic)
        late_arr = DIC_and_TIF().pix_dic_to_spatial_arr(late_dic)

        plt.figure()
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(pre_arr)
        plt.colorbar()
        plt.title('pre_arr')

        plt.figure()
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(early_arr)
        plt.title('early_arr')
        plt.colorbar()

        plt.figure()
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(late_arr)
        plt.title('late_arr')
        plt.colorbar()

        plt.show()

        pass


    def EGS_LGS_events(self,events_f,drought_index_f,outdir,growing_season_f):
        # outdir = self.this_class_arr + 'EGS_LGS_events\\'
        outf = outdir + 'EGS_LGS_events'
        T.mk_dir(outdir,force=True)
        # growing_season_f = Winter().this_class_arr + 'gen_grow_season_index\\growing_season_index.npy'
        # growing_season_f = Phenology_based_on_Temperature_NDVI().this_class_arr + 'growing_season_index.npy'
        growing_season_dic = T.load_npy(growing_season_f)
        ############### plot growing season spatial #######################
        # spatial_dic = {}
        # for pix in growing_season_dic:
        #     start = growing_season_dic[pix][0]
        #     spatial_dic[pix] = start
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        ############### plot growing season spatial #######################


        event_dic = T.load_npy(events_f)

        events_num = 0.
        for pix in event_dic:
            events = event_dic[pix]
            events_num += len(events)
        print 'events_num',events_num

        spei_dic = T.load_npy(drought_index_f)
        event_with_marks_dic = {}
        for pix in event_dic:
            if not pix in growing_season_dic:
                continue
            gs = growing_season_dic[pix]
            spei = spei_dic[pix]
            events = event_dic[pix]
            if len(events) > 0:
                event_with_marks = []
                for event in events:
                    if len(gs) == 12:
                        mark = 'tropical'
                    else:
                        min_indx = T.pick_min_indx_from_1darray(spei,event)
                        mon = min_indx % 12 + 1
                        p1 = gs[0] - 2
                        p2 = gs[0] - 1
                        if p1 <= 0:
                            p1 = p1 + 12
                        if p2 <= 0:
                            p2 = p2 + 12
                        e1,e2,e3,l1,l2,l3 = gs
                        all_mon = [p1,p2,e1,e2,e3,l1,l2,l3]
                        marks_gs = ['p1','p2','e1','e2','e3','l1','l2','l3']
                        if mon in all_mon:
                            mon_gs_indx = all_mon.index(mon)
                            mark = marks_gs[mon_gs_indx]
                        else:
                            mark = None
                    event_with_marks.append([mark, event])
                event_with_marks_dic[pix] = event_with_marks
        np.save(outf, event_with_marks_dic)
        pass

    def pick(self,f, outdir):
        # 前n个月和后n个月无极端干旱事件
        n = 24.
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        dic = T.load_npy(f)
        for pix in tqdm(dic,desc='picking {}'.format(f)):
            vals = dic[pix]
            mean = np.mean(vals)
            std = np.std(vals)
            threshold = mean - 2 * std
            # threshold = -1.5
            # threshold = np.quantile(vals, 0.05)
            event_dic,key = self.kernel_find_drought_period([vals,pix,threshold])
            if len(event_dic) == 0:
                continue
            events_4 = []
            for i in event_dic:
                level,drought_range = event_dic[i]
                events_4.append(drought_range)

            single_event = []
            for i in range(len(events_4)):
                if i - 1 < 0:  # 首次事件
                    if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(vals):  # 触及两边则忽略
                        continue
                    if len(events_4) == 1:
                        single_event.append(events_4[i])
                    elif events_4[i][-1] + n <= events_4[i + 1][0]:
                        single_event.append(events_4[i])
                    continue

                # 最后一次事件
                if i + 1 >= len(events_4):
                    if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(vals):
                        single_event.append(events_4[i])
                    break

                # 中间事件
                if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                    single_event.append(events_4[i])
            # print single_event
            # sleep(0.1)
            single_event_dic[pix] = single_event
            # for evt in single_event:
            #     picked_vals = T.pick_vals_from_1darray(vals,evt)
            #     plt.scatter(evt,picked_vals,c='r')
            # plt.plot(vals)
            # plt.show()
        np.save(outdir + 'single_events',single_event_dic)
        # spatial_dic = {}
        # for pix in single_event_dic:
        #     evt_num = len(single_event_dic[pix])
        #     if evt_num == 0:
        #         continue
        #     spatial_dic[pix] = evt_num
        # DIC_and_TIF().plot_back_ground_arr()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()


    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        key = params[1]
        threshold = params[2]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:# SPEI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []

        flag = 0
        events_dic = {}
        # 不取两个端点
        for i in events:
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                new_i.append(jj)
            # print(new_i)
            # exit()
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:
            # SPEI
            min_val = min(vals)
            if min_val < -99999:
                continue
            if min_val < threshold:
                level = 4
            # if -1 <= min_val < -.5:
            #     level = 1
            # elif -1.5 <= min_val < -1.:
            #     level = 2
            # elif -2 <= min_val < -1.5:
            #     level = 3
            # elif min_val <= -2.:
            #     level = 4
            else:
                level = 0


            events_dic[flag] = [level, new_i]
            # print(min_val)
            # plt.plot(vals)
            # plt.show()
        # for key in events_dic:
        #     # print key,events_dic[key]
        #     if 0 in events_dic[key][1]:
        #         print(events_dic[key])
        # exit()
        return events_dic, key


class Rs_Rc_Rt:

    def __init__(self):

        pass


    def run(self):
        # d = during the dry period
        # pre = 3 years previous dry period
        # post = post
        ndvi_dir = data_root + 'NDVI\\'

        rc = NDVI_d / NDVI_pre
        rc = NDVIG_post / NDVI_d
        rs = NDVI_post / NDVI_pre
        pass


def main():
    # 1通过NDVI 和 T的correlation 筛选有季节性的像素
    # Phenology_based_on_Temperature_NDVI().run()
    # Pick_drought_events().run()
    # 2计算rs rc rt
    Rs_Rc_Rt().run()

    pass


if __name__ == '__main__':

    main()
