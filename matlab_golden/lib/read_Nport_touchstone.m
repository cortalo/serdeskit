function [sch,schFreqAxis]=read_Nport_touchstone(touchstone_file,port_order)

%touchstone_file:  .sNp touchstone file to read
%port_order:  port reorder vector
%
%sch:  sparameter matrix
%schFreqAxis:  frequency axis

[file_path,root_name,extension]=fileparts(touchstone_file);
fid=fopen(touchstone_file);

%fetch number of ports from extension
num_ports=str2num(char(regexp(extension,'\d*','match')));

%Get option line
[optstr,opt_pos] = textscan(fid,'%s',1,'Delimiter','','CommentStyle','!');
optcell=textscan(optstr{1}{1},'%s');
optcell=optcell{1};
while isempty(optcell) || isempty(strfind(optcell{1},'#'))
    %Some touchstone files need this.  can't remember why now.  maybe lines
    %with whitespace but not empty but not commented
    [optstr,opt_pos] = textscan(fid,'%s',1,'Delimiter','','CommentStyle','!');
    optcell=textscan(optstr{1}{1},'%s');
    optcell=optcell{1};
end

%read the entire file
raw_read_data = textscan(fid,'%f %f %f %f %f %f %f %f %f','CollectOutput',true,'CommentStyle','!');
raw_column_data=raw_read_data{1};
fclose(fid);

%number of columns for 2D matrix
columns=num_ports*num_ports*2+1;

%find the frequency lines by searching for the right number of NaN
a=sum(isnan(raw_column_data),2);
if num_ports==3
    b=find(a==2);
elseif num_ports==1
    b=find(a==6);
else
    b=find(a==0);
end

num_freq=length(b);

%toss out the NaN and reshape into a 2D matrix
raw_input = raw_column_data.';
raw_input = raw_input(~isnan(raw_input));
raw_input = reshape(raw_input,columns,num_freq).';

%get the frequency mult
frequency_mult_text=optcell{2};
if(strcmpi(frequency_mult_text,'hz'))
    frequency_mult=1;
elseif(strcmpi(frequency_mult_text,'khz'))
    frequency_mult=1e3;
elseif(strcmpi(frequency_mult_text,'mhz'))
    frequency_mult=1e6;
elseif(strcmpi(frequency_mult_text,'ghz'))
    frequency_mult=1e9;
else
    error('Unsupported format for frequency multiplier %s',frequency_mult_text);
end

%get the RI/MA/DB format
format=optcell{4};
%get Z0
port_impedance=str2double(optcell(6:end))';


%grab frequency
raw_input(:,1)=raw_input(:,1)*frequency_mult;
Spar.F=raw_input(:,1);
Spar.F=transpose(Spar.F(:));


%transform data to real imaginary
%for 2.0 support, keep it in 2D form instead of 3D because we may need to process upper/lower sparse matrix definitions
if(strcmpi(format,'ri'))
    ri_data_2D=raw_input(:,2:2:end)+raw_input(:,3:2:end)*1i;
elseif(strcmpi(format,'ma'))
    mag_data=raw_input(:,2:2:end);
    rad_data=raw_input(:,3:2:end)*pi/180;
    ri_data_2D=mag_data.*cos(rad_data)+mag_data.*sin(rad_data)*1i;
elseif(strcmpi(format,'db'))
    mag_data=10.^(raw_input(:,2:2:end)/20);
    rad_data=raw_input(:,3:2:end)*pi/180;
    ri_data_2D=mag_data.*cos(rad_data)+mag_data.*sin(rad_data)*1i;
else
    error('Format %s is not supported.  Use RI MA or DB',format);
end



%transform to 3D
%allow for upper/lower matrix specification for touchstone 2.0 support
matrix_format=0;
if(matrix_format==0)
    %full
    for j=1:num_ports
        pre_out.sp(j,1:num_ports,:)=transpose(ri_data_2D( :,(j-1)*num_ports+1:j*num_ports));
    end
elseif(matrix_format==1)
    %upper
    used_ports=0;
    for j=1:num_ports
        stated_ports=num_ports-j+1;
        pre_out.sp(j,j:num_ports,:)=transpose(ri_data_2D( :,used_ports+1:used_ports+stated_ports));
        pre_out.sp(j:num_ports,j,:)=transpose(ri_data_2D( :,used_ports+1:used_ports+stated_ports));
        used_ports=used_ports+stated_ports;
    end
elseif(matrix_format==2)
    %lower
    used_ports=0;
    for j=1:num_ports
        stated_ports=j;
        pre_out.sp(j,1:j,:)=transpose(ri_data_2D( :,used_ports+1:used_ports+stated_ports));
        pre_out.sp(1:j,j,:)=transpose(ri_data_2D( :,used_ports+1:used_ports+stated_ports));
        used_ports=used_ports+stated_ports;
    end
else
    error('Matrix format is not supported.  Use Full, Lower, or Upper');
end


%check for swapping the 2 port matrix (required on 1.x spec)
two_port_swap=1;
if(num_ports==2 && two_port_swap==1)
    temp=pre_out.sp(1,2,:);
    pre_out.sp(1,2,:)=pre_out.sp(2,1,:);
    pre_out.sp(2,1,:)=temp;
end

Spar.S=pre_out.sp;
Spar.Z0=transpose(port_impedance(:));

if length(Spar.Z0)>1 || ~isequal(Spar.Z0,50)
    %issue error or warning?
end

%These operations sync up with COM style Spar matrix
%1:  put frequency as first dimension
sch=shiftdim(Spar.S,2);
%2:  reorder ports according to "ports" input
sch=sch(:,port_order,port_order);
schFreqAxis=Spar.F;
