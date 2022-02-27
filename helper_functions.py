def get_datafiles(data_dir, data, row_num, col_num):
    
    die = 'Die_01/'

    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_01 Not Found')
        data.append({})


    die = 'Die_07/'

    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_07 Not Found')
        data.append({})


    die = 'Die_13/'

    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_13 Not Found')
        data.append({})


    die = 'Die_15/'

    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_15 Not Found')
        data.append({})


    die = 'Die_17/'

    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_17 Not Found')
        data.append({})


    die = 'Die_25/'

    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_25 Not Found')
        data.append({})
        
        
    die = 'Die_39/'

    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '*_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_39 Not Found')
        data.append({})
        
    die = 'Die_19/'
    
    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '*_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_19 Not Found')
        data.append({})
        
    die = 'Die_49/'
    
    try:
        filename=glob.glob(data_dir+die+ '*' + str(row_num) + '*_col_' + str(col_num) + '_VW_sweep.pkl')
        data.append(pickle.load( open(filename[0],"rb")))
    except:
        print('Die_49 Not Found')
        data.append({})

        
def plot_VW(data, 
                chan = 1, 
                zoom_span = 5,
                prominence = 4.5,
                fit_order = 8,
                fsr = 15,
                ideal_peak_wl = 1550,
                index = 0,
                save = False):
        
        
        try: 
            fig = plt.figure( figsize = (9,7), dpi = 500 )
            widths = [1, 1, 0.1]
            heights = [1, 1]
            spec = fig.add_gridspec(ncols=3, 
                                     nrows=2, 
                                     width_ratios=widths,
                                     height_ratios=heights)




            colors = plt.cm.viridis(np.linspace(0,1,len(data['voltages'])))
            a0 = fig.add_subplot(spec[0,:-1]) #raw spectra
            a01 = fig.add_subplot(spec[:,2]) #voltage applied
            a1 = fig.add_subplot(spec[1,0]) #IV curve
            a2 = fig.add_subplot(spec[1,1]) #zoom window


            wl = np.array(data['wavelengths'][index])
            pw = np.array(data['powers'][index][chan - 1])

            ridx = np.isfinite(wl) & np.isfinite(pw)

            baseline = np.poly1d(np.polyfit(wl[ridx],pw[ridx],fit_order))

            pw_bs = pw - baseline(wl) + max(baseline(wl)) - max(pw)
            wl_per_args = wl[1] - wl[0]
            peaks ,_ = find_peaks(-pw_bs, prominence = prominence, distance = int(fsr/wl_per_args))

            idx = np.abs(wl[peaks] - ideal_peak_wl).argmin()
            center_peak = peaks[idx]

            span = int(0.5*zoom_span/wl_per_args)
            start_zoom, stop_zoom = center_peak - span, center_peak + span

            for i in range(len(data['wavelengths'])):
                wl = np.array(data['wavelengths'][i])
                pw = np.array(data['powers'][i][chan - 1])

                a0.plot(wl, pw, 
                        lw = 0.5, 
                        color = colors[i])

                a2.plot(wl[start_zoom:stop_zoom], 
                        pw[start_zoom:stop_zoom],
                        color = colors[i])


            norm = mpl.colors.Normalize(vmin=np.min(data['voltages']), 
                                        vmax=np.max(data['voltages']))
            cbar = a01.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                                       ax=a01, 
                                       fraction=1, pad=0.04,
                                       extend='both', 
                                       label = 'Voltage Applied')

            a01.axis('off')

            a0.set_title('Raw Spectra for Dev: ' + data['name'] + ' Measured: ' + data['datetime'].strftime('%b %d, %Y at %H:%M') )
            a0.set_xlabel('Wavelength (nm)')
            a0.set_ylabel('Power (dBm)')
            a0.xaxis.set_major_locator(MaxNLocator(18)) 

            a2.set_title('Zoom Window to Peak Nearest {:} nm'.format(ideal_peak_wl))
            a2.set_xlabel('\nWavelength (nm)')
            a2.set_ylabel('Power (dBm)')


            a1.plot(data['voltages'], data['currents']*1e3, '-o', color = 'purple')
            a1.set_title('IV Curve')
            a1.set_xlabel('Voltage (V)')
            a1.set_ylabel('Current (mA)')

            fig.tight_layout()
            if save:
                fig.savefig(directory + data['name'] + '_VWSweep.png', 
                            dpi = 500)
                plt.close()
            else:
                plt.show()
        
        except:
            print('Empty data')


def extract_peaks(data, 
                    chan = 1, 
                    zoom_span = 0.75,
                    prominence = 4.5,
                    fit_order = 8,
                    fsr = 15,
                    ideal_peak_wl = 1550,
                    exclude = False):
    
    try:
        peak_wl = np.array([])
        peak_pw = np.array([])
        peak_width = np.array([])
        peak_depth = np.array([])

        for i in range(len(data['wavelengths'])):
            wl = np.array(data['wavelengths'][i])
            pw = np.array(data['powers'][i][chan - 1])

            ridx = np.isfinite(wl) & np.isfinite(pw)

            baseline = np.poly1d(np.polyfit(wl[ridx],pw[ridx],fit_order))

            pw_bs = pw - baseline(wl) + max(baseline(wl)) - max(pw)
            wl_per_args = wl[1] - wl[0]
            peaks ,_ = find_peaks(-pw_bs, prominence = prominence, distance = int(fsr/wl_per_args))

            idx = np.abs(wl[peaks] - ideal_peak_wl).argmin()
            center_peak = peaks[idx]

            span = int(0.5*zoom_span/wl_per_args)
            start_zoom, stop_zoom = center_peak - span, center_peak + span

            wl_left = wl[start_zoom:center_peak]
            wl_right = wl[center_peak:stop_zoom]
            pw_left = pw[start_zoom:center_peak]
            pw_right = pw[center_peak:stop_zoom]

            peak_wl = np.append(peak_wl, wl[center_peak])
            peak_pw = np.append(peak_pw, pw[center_peak])
            peak_depth = np.append(peak_depth, peak_prominences(-pw_bs, center_peak.flatten(), wlen=span*2)[0])

            peak_half = (peak_pw[i] + (peak_depth[i]/2))
            peak_left = wl_left[np.abs(pw_left - peak_half).argmin()]
            peak_right = wl_right[np.abs(pw_right - peak_half).argmin()]
            peak_width = np.append(peak_width, peak_right - peak_left)
        
        if(exclude == False):
            return peak_wl, peak_pw, peak_depth, peak_width
        else:
            return np.zeros(11), np.full(11, np.inf), np.zeros(11), np.zeros(11)
            
    
    except:
        print('Empty data')
        return np.zeros(11), np.full(11, np.inf), np.zeros(11), np.zeros(11)


def plot_resonances(data, peak, ave_res, row_num, col_num):
    
    # plot resonances, peak widths, and peak depths
    for i in range(0, len(peak), 6):
        
        fig = plt.figure( figsize = (20,14), dpi = 100 )
        fig.suptitle('row_' + str(row_num) + '_col_' + str(col_num) + ': ' + str(ave_res[int(i/6)]) + ' nm resonance')
        
        ax = fig.add_subplot(211)
        ax.set_title('Resonance and peak width as voltage is swept from -2.0V to 0.5V')
        ax.set_ylabel('Wavelength (nm)')
        ax.set_xlabel('Voltage (V)')
        
        ax2 = fig.add_subplot(212)
        ax2.set_title('Peak depth as voltage is swept from -2.0V to 0.5V')
        ax2.set_ylim(0,40)
        ax2.set_ylabel('Power (dBm)')
        ax2.set_xlabel('Voltage (V)')
        
        colors = ['r', 'g', 'b', 'orange', 'purple', 'grey', 'gold']
        dice = ['Die_01', 'Die_07', 'Die_13', 'Die_15', 'Die_17', 'Die_25', 'Die_37']
        
        v, r, w, d = [], [], [], []
        
        for j in range(6):
            
            try:
                voltages = data[j]['voltages']
                v.append(voltages[np.abs(peak[i+j,1]) != np.inf])
                r.append(peak[i+j,0][np.abs(peak[i+j,1]) != np.inf])
                w.append(peak[i+j,3][np.abs(peak[i+j,1]) != np.inf])
                d.append(peak[i+j,2][np.abs(peak[i+j,1]) != np.inf])

                ax.plot(v[j], r[j], color=colors[j], label=dice[j] + ' resonance')
                ax.plot(v[j], w[j], '--', color=colors[j], label=dice[j] + ' peak width')
                ax.legend(loc='upper right');

                ax2.plot(v[j], d[j], color=colors[j], label=dice[j] +' peak depth')
                ax2.legend(loc='upper right');
            except:
                voltages = np.zeros(len(peak[i+j,0]))
                v.append(voltages[np.abs(peak[i+j,1]) != np.inf])
                r.append(peak[i+j,0][np.abs(peak[i+j,1]) != np.inf])
                w.append(peak[i+j,3][np.abs(peak[i+j,1]) != np.inf])
                d.append(peak[i+j,2][np.abs(peak[i+j,1]) != np.inf])

                ax.plot(v[j], r[j], color=colors[j], label=dice[j] + ' resonance')
                ax.plot(v[j], w[j], '--', color=colors[j], label=dice[j] + ' peak width')
                ax.legend(loc='upper right');

                ax2.plot(v[j], d[j], color=colors[j], label=dice[j] +' peak depth')
                ax2.legend(loc='upper right');
        
        fig.savefig('figures/r' + str(row_num) + '_c' + str(col_num) + '_' + str(ave_res[int(i/6)]) + 'nm.png')
        
        
def plot_average_resonances(data, peak, ave_res, row_num, col_num):
    
    # plot resonances, peak widths, and peak depths
    for i in range(0, len(peak), 6):
        
        fig = plt.figure( figsize = (20,14), dpi = 100 )
        fig.suptitle('row_' + str(row_num) + '_col_' + str(col_num) + ': ' + str(ave_res[int(i/6)]) + ' nm resonance')
        
        ax = fig.add_subplot(211)
        ax.set_title('Average resonance and peak width')
        ax.set_ylabel('Wavelength (nm)')
        ax.set_xlabel('Voltage (V)')
        
        ax2 = fig.add_subplot(212)
        ax2.set_title('Average peak depth')
        ax2.set_ylim(0,40)
        ax2.set_ylabel('Power (dBm)')
        ax2.set_xlabel('Voltage (V)')
        
        v, r, w, d = [], [], [], []
        
        for j in range(6):
            
            try:
                voltages = data[j]['voltages']
                voltages[np.abs(peak[i+j,1]) == np.inf] = np.nan
                v.append(voltages)
                
                resonances = peak[i+j,0]
                resonances[np.abs(peak[i+j,1]) == np.inf] = np.nan
                r.append(resonances)
                
                widths = peak[i+j,3]
                widths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                w.append(widths)
                
                depths = peak[i+j,2]
                depths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                d.append(depths)
            except:
                voltages = np.zeros(len(peak[i+j,0]))
                voltages[np.abs(peak[i+j,1]) == np.inf] = np.nan
                v.append(voltages)
                
                resonances = peak[i+j,0]
                resonances[np.abs(peak[i+j,1]) == np.inf] = np.nan
                r.append(resonances)
                
                widths = peak[i+j,3]
                widths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                w.append(widths)
                
                depths = peak[i+j,2]
                depths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                d.append(depths)
        
        v_ave = np.nanmean((v), axis=0)
        r_ave = np.nanmean((r), axis=0)
        w_ave = np.nanmean((w), axis=0)
        d_ave = np.nanmean((d), axis=0)
        
        ax.plot(v_ave, r_ave, color='orangered', lw=3, label='average resonance')
        ax.plot(v_ave, w_ave, '--', color='orangered', lw=3, label='average peak width') 
        ax.legend(loc='upper right');
        
        ax2.plot(v_ave, d_ave, color='orangered', lw=3, label='average peak depth')
        ax2.legend(loc='upper right');
        
        v_std = np.nanstd((v), axis=0)
        r_std = np.nanstd((r), axis=0)
        w_std = np.nanstd((w), axis=0)
        d_std = np.nanstd((d), axis=0)
        
        ax.fill_between(v_ave, r_ave-r_std,r_ave+r_std, color='orangered', alpha=.1)
        ax.fill_between(v_ave, w_ave-w_std,w_ave+w_std, color='orangered', alpha=.1)
        
        ax2.fill_between(v_ave, d_ave-d_std,d_ave+d_std, color='orangered', alpha=.1)
        
        fig.savefig('figures/r' + str(row_num) + '_c' + str(col_num) + '_' + str(ave_res[int(i/6)]) + 'nm_ave.png')

        
def plot_res(
    data, 
    peak, 
    ave_res, 
    row_num, 
    col_num, 
    num_die=6,
    peak0=[],
    num_die0=3,
    colors = ['r', 'g', 'b', 'orange', 'purple', 'grey', 'black', 'blue', 'cyan'],
    dice = ['Die_01', 'Die_07', 'Die_13', 'Die_15', 'Die_17', 'Die_25', 'Die-39 (from Q4 Condition #1 - 3.6E13 Doping)', 'Die-19 (from Q3 Condition #2 - 3.6E13 Doping)', 'Die-49 (from Q4 Condition #2 6.6E13)']):
    
    # generate plots for each peak
    for i in range(0, len(peak), num_die):
        
        fig = plt.figure( figsize=(20,14), dpi=100 )
        fig.suptitle('row_' + str(row_num) + '_col_' + str(col_num) + ': ' + str(ave_res[i//num_die]) + ' nm resonance')
        
        ax1 = fig.add_subplot(211)
        ax1.set_title('Resonance and peak width as voltage is swept from -2.0V to 0.5V')
        ax1.set_ylabel('Wavelength (nm)')
        ax1.set_xlabel('Voltage (V)')
        
        ax2 = fig.add_subplot(212) 
        ax2.set_title('Peak depth as voltage is swept from -2.0V to 0.5V')
        ax2.set_ylim(0,40)
        ax2.set_ylabel('Power (dBm)')
        ax2.set_xlabel('Voltage (V)') 
        
        v, r, w, d = [], [], [], []

        # save average resonance, peak width, and peak depth data for each die
        for j in range(num_die):      
            try:
                voltages = data[j]['voltages']
                voltages[np.abs(peak[i+j,1]) == np.inf] = np.nan
                v.append(voltages)
                
                resonances = peak[i+j,0]
                resonances[np.abs(peak[i+j,1]) == np.inf] = np.nan
                r.append(resonances)
                
                widths = peak[i+j,3]
                widths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                w.append(widths)
                
                depths = peak[i+j,2]
                depths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                d.append(depths)
            except:
                voltages = np.zeros(len(peak[i+j,0]))
                voltages[np.abs(peak[i+j,1]) == np.inf] = np.nan
                v.append(voltages)
                
                resonances = peak[i+j,0]
                resonances[np.abs(peak[i+j,1]) == np.inf] = np.nan
                r.append(resonances)
                
                widths = peak[i+j,3]
                widths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                w.append(widths)
                
                depths = peak[i+j,2]
                depths[np.abs(peak[i+j,1]) == np.inf] = np.nan
                d.append(depths)
        
        # plot data for each die except comparison
        v_ave = np.nanmean((v), axis=0)
        r_ave = np.nanmean((r), axis=0)
        w_ave = np.nanmean((w), axis=0)
        d_ave = np.nanmean((d), axis=0)
        
        ax1.plot(v_ave, r_ave, color='orangered', lw=3, label='Average resonance from Q2 Cond #1 - 2.4E13 Doping')
        ax1.plot(v_ave, w_ave, '--', color='orangered', lw=3, label='Average peak from Q2 Cond #1 - 2.4E13 Doping') 
        ax2.plot(v_ave, d_ave, color='orangered', lw=3, label='Average peak depth from Q2 Cond #1 - 2.4E13 Doping')
        
        # show standard deviations
        v_std = np.nanstd((v), axis=0)
        r_std = np.nanstd((r), axis=0)
        w_std = np.nanstd((w), axis=0)
        d_std = np.nanstd((d), axis=0)
        
        ax1.fill_between(v_ave, r_ave-r_std,r_ave+r_std, color='orangered', alpha=.1)
        ax1.fill_between(v_ave, w_ave-w_std,w_ave+w_std, color='orangered', alpha=.1)
        ax2.fill_between(v_ave, d_ave-d_std,d_ave+d_std, color='orangered', alpha=.1)
        
        # plot data for comparison die
#         voltages0 = data[num_die]['voltages']
#         v0 = voltages0[np.abs(peak0[i//num_die,1]) != np.inf]
#         r0 = peak0[i//num_die,0][np.abs(peak0[i//num_die,1]) != np.inf]
#         w0 = peak0[i//num_die,3][np.abs(peak0[i//num_die,1]) != np.inf]
#         d0 = peak0[i//num_die,2][np.abs(peak0[i//num_die,1]) != np.inf]
        
#         ax1.plot(v0, r0, color=colors[num_die], label=dice[num_die] + ' resonance')
#         ax1.plot(v0, w0, '--', color=colors[num_die], label=dice[num_die] + ' peak width')
#         ax2.plot(v0, d0, color=colors[num_die], label=dice[num_die] +' peak depth')
        
#         ax1.legend(loc='upper right');
#         ax2.legend(loc='upper right');
        
        v0, r0, w0, d0 = [], [], [], []

        for j in range(num_die0):

            try:
                voltages0 = data[num_die+j]['voltages']
                v0.append(voltages0[np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])
                r0.append(peak0[i//num_die*num_die0+j,0][np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])
                w0.append(peak0[i//num_die*num_die0+j,3][np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])
                d0.append(peak0[i//num_die*num_die0+j,2][np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])

                ax1.plot(v0[j], r0[j], color=colors[num_die+j], label=dice[num_die+j] + ' resonance')
                ax1.plot(v0[j], w0[j], '--', color=colors[num_die+j], label=dice[num_die+j] + ' peak width')
                ax2.plot(v0[j], d0[j], color=colors[num_die+j], label=dice[num_die+j] +' peak depth')
            except:
                voltages0 = np.zeros(len(peak0[i//num_die*num_die0+j,0]))
                v0.append(voltages0[np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])
                r0.append(peak0[i//num_die*num_die0+j,0][np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])
                w0.append(peak0[i//num_die*num_die0+j,3][np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])
                d0.append(peak0[i//num_die*num_die0+j,2][np.abs(peak0[i//num_die*num_die0+j,1]) != np.inf])

                ax1.plot(v0[j], r0[j], color=colors[num_die+j], label=dice[num_die+j] + ' resonance')
                ax1.plot(v0[j], w0[j], '--', color=colors[num_die+j], label=dice[num_die+j] + ' peak width');
                ax2.plot(v0[j], d0[j], color=colors[num_die+j], label=dice[num_die+j] +' peak depth')
        
        ax1.legend(loc='upper right');
        ax2.legend(loc='upper right');

        fig.savefig('figures/r' + str(row_num) + '_c' + str(col_num) + '_' + str(ave_res[i//num_die]) + 'nm_ave.png')