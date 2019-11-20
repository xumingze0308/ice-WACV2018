fn_dir = '/l/vision/v7/mx6/data/CReSIS/images-old';
to_dir = '/l/vision/v7/mx6/data/CReSIS/slices_mat_64x64';

% Compile Cpp files
mex -largeArrayDims fuse.cpp

% img_01: right looking (positive theta)
% img_02: nadir looking (theta == 0)
% img_03: left looking (negative theta)
nadir_img = 2;

dates = {'20140325_05', '20140325_06', '20140325_07', '20140401_03', '20140506_01'};
for date = dates
    date = char(date);

    frame_root = [fn_dir '/' date];
    to_frame_root = [to_dir '/' date];
    mkdir(to_frame_root);

    frame_names = dir(frame_root);
    for i = 1:length(frame_names)
        if contains(frame_names(i).name, char('img_01'))
            file_name = strsplit(frame_names(i).name, '_');
            file_name = char(file_name(6));
            file_name = strsplit(file_name, '.');
            file_name = file_name(1);
            to_file_root = [char(to_frame_root) '/' char(file_name)];
            mkdir(to_file_root);

            % Load Data
            mdata = {};
            for img=1:3
                fn = fullfile(frame_root, strrep(frame_names(i).name,'img_01',[char('img_') num2str(img,'%02.f')]))
                mdata{img} = load(fn);
            end

            % Check for twtt DEM data
            if ~isfield(mdata{1},'twtt')
                Nx = length(mdata{1}.GPS_time);
                Nsv = length(mdata{1}.param_combine.array_param.theta);
                mdata{1}.twtt = NaN*zeros(Nsv,Nx);
            end

            % Interpolate images onto a common propagation time axis
            for img = [1 3]
                mdata{img}.Data = interp1(mdata{img}.Time,mdata{img}.Data,mdata{nadir_img}.Time);
                mdata{img}.Topography.img = interp1(mdata{img}.Time,mdata{img}.Topography.img,mdata{nadir_img}.Time);
                mdata{img}.Time = mdata{nadir_img}.Time;
            end

            % Convert Surface and Bottom variables from propagation time into image pixels
            Surface_bin = interp1(mdata{1}.Time, 1:length(mdata{1}.Time), mdata{1}.Surface);
            Surface_Mult_bin = interp1(mdata{1}.Time, 1:length(mdata{1}.Time), 2 * mdata{1}.Surface);
            Bottom_bin = interp1(mdata{1}.Time, 1:length(mdata{1}.Time), mdata{1}.Bottom);
            twtt_bin = interp1(mdata{1}.Time, 1:length(mdata{1}.Time), mdata{1}.twtt);
            Bottom_bin(isnan(Bottom_bin)) = 0;
            twtt_bin(isnan(twtt_bin)) = 0;

            %% Automated labeling section
            % =========================================================================
            % Specify which range lines to browse
            skip = 1;
            rlines = 1:skip:size(mdata{1}.Topography.img,3);

            for rline = rlines
                fusion = fuse(double(db(mdata{1}.Topography.img(:,:,rline))), ...
                    double(db(mdata{2}.Topography.img(:,:,rline))), ...
                    double(db(mdata{3}.Topography.img(:,:,rline))));
                % fusion(fusion>(min(fusion(:))+20)) = min(fusion(:))+20;
                fusion(fusion>27) = 27;
                fusion = reshape(fusion, size(db(mdata{1}.Topography.img(:,:,rline))));
                fusion = imresize(fusion, [64, 64]);
                fusion = mat2gray(fusion);
                outfile = char(strcat(to_frame_root, '/', file_name, '/', num2str(rline,'%05.f'), '.mat'))
                save(outfile, 'fusion');
            end
        end
    end
end
