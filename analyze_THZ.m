% Give time resolution
global dt 
dt = 0.05;


filenames=["QW_4K_QWP_number_front_0.00T.txt",
    "QW_4K_QWP_number_front_-0.50T.txt",
    "QW_4K_QWP_number_front_-1.00T.txt",
    "QW_4K_QWP_number_front_-1.50T.txt",
    "QW_4K_QWP_number_front_-2.00T.txt",
    "QW_4K_QWP_number_front_-2.50T.txt",
    "QW_4K_QWP_number_front_-3.00T.txt",
    "QW_4K_QWP_number_front_-3.50T.txt",
    "QW_4K_QWP_number_front_-4.00T.txt",
    "QW_4K_QWP_number_front_-4.50T.txt",
    "QW_4K_QWP_number_front_-5.00T.txt"];

% Plot THz time_domain data
figure
hold on
plot_raw_data_tdomain(filenames,1,0) % 0 Tesla data
plot_raw_data_tdomain(filenames,6,0) 
plot_raw_data_tdomain(filenames,11,0)


xlabel('Time(ps)')
ylabel('Signal(mV)')
legend('0 Tesla','2.5 Tesla','5 Tesla')
hold off

% Plot THz freq_domain data
figure
hold on
fft_thz(filenames,1,2.5,1,0); % 0 Tesla data
fft_thz(filenames,5,2.5,1,0); % 2 Tesla data
fft_thz(filenames,7,2.5,1,0); % 3 Tesla data

xlabel('Frequency(THz)')
ylabel('Signal(V)')
legend('0 Tesla','2 Tesla','3 Tesla')
hold off

% Plot THz absorption
figure
hold on
fft_thz(filenames,3,2.5,2,0); % 1 Tesla data
fft_thz(filenames,5,2.5,2,0); % 2 Tesla data
fft_thz(filenames,7,2.5,2,0); % 3 Tesla data
fft_thz(filenames,9,2.5,2,0); % 4 Tesla data
fft_thz(filenames,11,2.5,2,0); % 5 Tesla data

xlabel('Frequency(THz)')
ylabel('Signal/Reference(V/V)')
legend('1 Tesla','2 Tesla','3 Tesla','4 Tesla','5 Tesla')
hold off

B_field = abs([0,-0.5,-1.,-1.5,-2.,-2.5,-3.,-3.5,-4.,-4.5,-5.]);
absorption_peak_freq = [];


for i=1:length(B_field)

peak = fft_thz(filenames,i,2.5,0,0);
absorption_peak_freq = [absorption_peak_freq,peak];

end

figure
hold on;

scatter(B_field,absorption_peak_freq)


P = polyfit(B_field,absorption_peak_freq,1);
slope = P(1);
intercept = P(2);
yfit = P(1)*B_field+P(2);  % P(1) is the slope and P(2) is the intercept

plot(B_field,yfit,'r-.')

xlabel('Magnetic field(T)')
ylabel('peak frquency(THz)')

hold off

e_charge = 1.6e-19;
m_electron = 9.11e-31;
m_eff = e_charge/(2*pi*slope*1e12)/m_electron

% Plot THz time_domain data after subtraction with 0-field data
figure
hold on
plot_raw_data_tdomain(filenames,1,1) % 0 Tesla data
plot_raw_data_tdomain(filenames,3,1) 
plot_raw_data_tdomain(filenames,5,1)
plot_raw_data_tdomain(filenames,7,1) 
plot_raw_data_tdomain(filenames,9,1)


xlabel('Time(ps)')
ylabel('Superradiant CR Signal(mV)')
legend('0 Tesla','1 Tesla','2 Tesla','3 Tesla','4 Tesla')
hold off

% Plot THz freq_domain data
figure
ts_decay = [];
hold on
for i=[3,4,5,6,7,8,9]
[f_peak,f,amp_f] = fft_thz(filenames,i,2.5,1,1);;
[w0,sig,t_decay] = fit_lorentzian(f,amp_f);
ts_decay = [ts_decay,t_decay];
end

xlabel('Frequency(THz)')
ylabel('Superradiant CR Amp(V)')
legend('1 Tesla','1.5 Tesla','2 Tesla','2.5 Tesla','3 Tesla','3.5 Tesla','4 Tesla')
hold off

figure
plot([1,1.5,2,2.5,3,3.5,4],ts_decay)
xlabel('B(T)')
ylabel('SR Decay Time(ps)')


function data_out= read_data(filename)

    t =readtable(filename);
    data = table2array(t);
    data_out = data.';

end


function plot_raw_data_tdomain(filenames,idx,substract_zero_field)
    signal_t = read_data(filenames(idx));
    dt = 0.05;
    signal = signal_t;
    time = linspace(0,dt*(length(signal)-1),length(signal));
    
    if substract_zero_field==1
        signal0 = read_data(filenames(1));
        plot(time,(signal-signal0)*1e3)
    else
        plot(time,signal*1e3)
    end

end

function [f_peak,f,amp_f] = fft_thz(filenames,idx,freq_max,plosignal_t,plot_deviation)
    global dt 
    signal_t0 = read_data(filenames(1));
    signal_ti = read_data(filenames(idx));
    amp_f0 = fft(signal_t0);
    signal_delta = [];
    if plot_deviation==1
        for i=1:length(signal_t0)
            signal_delta = [signal_delta,(signal_ti(i)-signal_t0(i))];
        end
        amp_f = fft(signal_delta);
    else
        amp_f = fft(signal_ti);
    end
    

    fs = 1/dt;
    f = (0:length(amp_f)-1)*fs/length(amp_f);
    
    absorption = abs(amp_f./(amp_f0));
    
    f_idx_max = int16(freq_max/(f(2)-f(1)));
    if plosignal_t==1
        plot(f(1:f_idx_max),abs(amp_f(1:f_idx_max)));
    elseif plosignal_t==2
        plot(f(1:f_idx_max),absorption(1:f_idx_max));
    end

    [abs_value,abs_peak_idx] = min(absorption(1:f_idx_max));
    f_peak = f(abs_peak_idx);
end

function [w0,sig,t_decay] = fit_lorentzian(x,y)
    [y0,idx0] = max(abs(y));
    w0 = x(idx0);
    function y_fit = lorentzian(x,y0,sig)
        y_fit = [];
        for i=1:length(x)
            y_fit = [y_fit,y0/(1+((x(i)-x(idx0))/sig)^2) ];
        end
    end
    
    function err = error(sig,x,y)
        y_fit = lorentzian(x,y0,sig);
        err=sum((abs(y)-y_fit).*(abs(y)-y_fit));
    end
    fun = @(sig)error(sig,x,y);
    sig = fminbnd(fun,0,w0);
    t_decay = 1/sig;
end