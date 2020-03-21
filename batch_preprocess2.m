eeglab
set_file=dir('D:\\Files\\SJTU\\Study\\MME_Lab\\Teacher_Lu\\click_number\\eeg\\segmentation\\preprocess1\\*.set');

for n = 1 : length(set_file)
     i = find('.'==set_file(n).name);
    file_name = set_file(n).name(1:i-1);
    EEG = pop_loadset('filename',[file_name,'.set'],'filepath',[set_file(n).folder, '\']);
    EEG = eeg_checkset( EEG );
    
    bad_channel = importdata(['D:\\Files\\SJTU\\Study\\MME_Lab\\Teacher_Lu\\click_number\\eeg\\segmentation\\bad_channel\\', file_name, '.txt']);
    bad_channel = str2num(bad_channel{1});
    pca = 29 - length(bad_channel);
    if pca < 20
        continue;
    end
    
    EEG = pop_interp(EEG, bad_channel, 'spherical');
    EEG = eeg_checkset( EEG );
    EEG = eeg_checkset( EEG );
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'pca',pca,'interrupt','on');
	EG = eeg_checkset( EEG );
    EEG = pop_iclabel(EEG,'Default');
    EEG = eeg_checkset( EEG );
    EEG = pop_icflag(EEG, [NaN NaN;0.5 1;0.5 1;0.7 1;0.7 1;0.7 1;0 0.4]);
    EEG = eeg_checkset( EEG );
    com = find(EEG.reject.gcompreject);
    if length(com) > 9
        continue;
    end
    EEG = pop_subcomp( EEG, com, 0);
    EEG = eeg_checkset( EEG );
    
    EEG = pop_saveset( EEG, 'filename',[file_name,'.set'],'filepath','D:\\Files\\SJTU\\Study\\MME_Lab\\Teacher_Lu\\click_number\\eeg\\segmentation\\preprocess2\\');
    EEG = eeg_checkset( EEG );
end
