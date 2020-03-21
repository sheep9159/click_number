eeglab
eeg_file=dir('D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\segmentation\raw_data\*.eeg');

for n = 1 : length(eeg_file)
     i = find('.'==eeg_file(n).name);
    file_name = eeg_file(n).name(1:i-1);
    data_length = eeg_file(n).bytes / (4 * 35);
    EEG = pop_loadbv('D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\segmentation\raw_data\', [file_name,'.vhdr'], [1 data_length], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]);
    EEG = eeg_checkset( EEG );
    EEG = pop_select( EEG, 'nochannel',{'ACC30' 'ACC31' 'ACC32' 'Packet' 'TRIGGER'});
    EEG = eeg_checkset( EEG );
    EEG=pop_chanedit(EEG, 'lookup','Standard-10-5-Cap385.sfp');
    EEG = eeg_checkset( EEG );
    EEG = pop_reref( EEG, 30);
    EEG = eeg_checkset( EEG );
    EEG = pop_eegfiltnew(EEG, 'locutoff',0,'hicutoff',42);
    EEG = eeg_checkset( EEG );
    EEG = pop_eegfiltnew(EEG, 'locutoff',45,'hicutoff',55,'revfilt',1);
    EEG = eeg_checkset( EEG );
    EEG = pop_rmbase( EEG, [],[]);
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',[file_name,'.set'],'filepath','D:\\Files\\SJTU\\Study\\MME_Lab\\Teacher_Lu\\click_number\\eeg\\segmentation\\preprocess1\\');
    EEG = eeg_checkset( EEG );
end