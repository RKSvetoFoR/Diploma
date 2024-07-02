// main.cpp for Projectile problem

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define __CL_ENABLE_EXCEPTIONS
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <ctime>
#include <CL/cl2.hpp>
//using namespace std;

#define _max_(X,Y) ((X)>(Y)?(X):(Y))
#define _min_(X,Y) ((X)<(Y)?(X):(Y))
#define _norm_(x) (x)

const int N = 110;
const double H = (1.0E-3) * 2 ;
const double DT = (H * H) / 2;
const double TMAX = 1.E-1;
const double DTH = DT / H;
const double pi_3 = M_PI;
const int save_step = 10000;
const double TAUH = DT / 100;//!!!
const int nsteph = 100;//!!!


const int nMat = 7;//*<количество веществ //!!!!!
const int OsnV = 4; //*<количество основных веществ //!!!!!
const int nst = 5; //*<количество стадий //!!!!!
const double P0 = 1.01325E+5;
const double temp0 = 950.0;
const double temp_in = 950.0;
const double gR = 8.314472;	//*< универсальная газовая постоянная
const double u_g = 0.1;
const double A[nst] = { 1.e18,1.e10,3.16e16,4.47e9,1.e10 };//*<предэкспоненциальный //!!!!!
const double En[nst] = { 360000.0,50000.0,170000.0,40000.0,8400.0 };//*<энергия активации //!!!!!



double* ro, * u, * rh, * pi, * temp, **ry;//N+2
double* ro_int, * u_intt, * rh_int, ** ry_int, * S;
double* ro_old, * u_old, * rh_old, * temp_old, ** ry_old;
double* y, * y_m, * yl, * yr, * fry, * urs; //nMat

double* Mm, * h0, * ML0, * KP0, * Sig, * ek; //nMat
double* hii, * hir, * hil, * Dm, * Dml, * Dmr; //nMat
double* uu, * uu_, * uu_s, * phi, * psi, * Ri; //nMat

int         n_TCP; //!< количество слагаемых в Cp
double** k_TCP; //!< коэффициенты в выражениях Cp 

double** TD, ** WD, ** Dij;

double* DF, * DFr, * DFl;

cl::Program program;    // The program that will run on the device.
cl::Context context;    // The context which holds the device.
cl::Device device;      // The device where the kernel will run.



/**
 * Чтение файла ядра OpenCL
 */
void initializeDevice();
/**
 * Выбор устройства OpenCL
 */
cl::Device getDefaultDevice();
/**
 * Заполнение буферов OpenCL
 */
void mem_alloc();
/**
 * Освобождение памяти
 */
void mem_free();
/**
 * Начальные данные
 */
void init();
/**
 * Уравнение состояния смеси идеальных газов
 */
void urs_mixt(double* y_m, double h, double p, double fT, int flag,
	double& tmp, double& gamma, double& r, double& h_);
/**
 * Теплоемкость при постоянном давлении
 */
double Calc_CP(int iM, double fT);

/**
 * Вычисление температуры итерациями по Ньютону
 */
void Newton_Method(double e, double* y_m, double tmp1,
	double& tmp);
/**
 * Вычисление энтальпии компонент
 */
void calc_h(double tmp,
	double hii[nMat]);


cl::Device getDefaultDevice() 
{

	/**
	 * Search for all the OpenCL platforms available and check
	 * if there are any.
	 * */

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);



	if (platforms.empty()) {
		std::cerr << "No platforms found!" << std::endl;
		exit(1);
	}

	/**
	 * Search for all the devices on the first platform and check if
	 * there are any available.
	 * */

	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	if (devices.empty()) {
		std::cerr << "No devices found!" << std::endl;
		exit(1);
	}

	/**
	 * Return the first device found.
	 * */

	cl::Device device = devices.front();

	// Get and print the OpenCL version
	std::string opencl_version;
	device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &opencl_version);
	std::cout << "OpenCL version: " << opencl_version << std::endl;

	// Get and print the device type
	cl_device_type device_type;
	device.getInfo(CL_DEVICE_TYPE, &device_type);
	std::string type;
	switch (device_type) {
	case CL_DEVICE_TYPE_CPU:
		type = "CPU";
		break;
	case CL_DEVICE_TYPE_GPU:
		type = "GPU";
		break;
	case CL_DEVICE_TYPE_ACCELERATOR:
		type = "Accelerator";
		break;
	default:
		type = "Unknown";
	}
	std::cout << "Device type: " << type << std::endl;

	// Get and print the device name
	std::string device_name;
	device.getInfo(CL_DEVICE_NAME, &device_name);
	std::cout << "Device name: " << device_name << std::endl;

	//	std::cout << "Work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
	return device;
}

void initializeDevice() 
{

	/**
	 * Select the first available device.
	 * */

	device = getDefaultDevice();

	/**
	 * Read OpenCL kernel file as a string.
	 * */

	std::fstream input;
	input.open("kernel.cl", std::ios::in);
	std::string kernel_code = "";

	if (input.is_open()) {
		std::string line;

		while (getline(input, line)) {
			kernel_code += line;
		}
	}
	cl::Program::Sources sources;

	/**
	 * Compile kernel program which will run on the device.
	 * */
	sources.push_back({ kernel_code.c_str(),kernel_code.length() }); std::ifstream kernel_file("kernel.cl");
	context = cl::Context(device);
	program = cl::Program(context, sources);

	auto err = program.build();
	if (err != CL_BUILD_SUCCESS) {
		std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
			<< "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}

}

void init()
{
	double r, p, e, gam, tmp, h_;

	// читаем данные о коэффициентах для расчёта Сp
	int n;
	FILE* f = fopen("Cp.dat", "r");
	fscanf(f, "%d %d", &n, &n_TCP);
	k_TCP = new double* [nMat];
	for (int i = 0; i < nMat; i++) k_TCP[i] = new double[n_TCP];
	for (int i = 0; i < nMat; i++)
	{
		for (int j = 0; j < n_TCP; j++) fscanf(f, "%lf", &k_TCP[i][j]);
	}
	fclose(f);
	for (int j = 0; j < 10; j++) {
		urs[j] = 0.;
	}
	Mm[0] = 0.03007012; h0[0] = -8.386329e4 / Mm[0];//ethane
	Mm[1] = 0.02805418; h0[1] = 5.24554e4 / Mm[1];//ethylene
	Mm[2] = 0.00201594; h0[2] = 0.0;   //hydrogen
	Mm[3] = 0.01604303; h0[3] = -7.489518e4 / Mm[3]; //methane
	Mm[4] = 0.01503506; h0[4] = 1.456861e5 / Mm[4]; //CH3*
	Mm[5] = 0.02906215; h0[5] = 1.172138e5 / Mm[5];   //C2H5*
	Mm[6] = 0.00100797; h0[6] = 2.17965e5 / Mm[6];//H* 

	//0-C2H6; 1-C2H4; 2-H2; 3-CH4; 4-CH3*; 5-C2H5*; 6-H*; 
	KP0[0] = 0.017839;  ML0[0] = 8.5911e-6;//ethane
	KP0[1] = 0.017249;  ML0[1] = 9.4514e-6;//ethylene
	KP0[2] = 0.17244;   ML0[2] = 8.3943e-6;//hydrogen
	KP0[3] = 0.030845;  ML0[3] = 1.0352e-5;//methane
	KP0[4] = 0.0454;    ML0[4] = 1.72e-5;//CH3*
	KP0[5] = 0.0454;    ML0[5] = 1.72e-5;//C2H5*
	KP0[6] = 0.2316;    ML0[6] = 7.49e-6;//H* 


	Sig[0] = 3.512;  ek[0] = 139.8;//ethane
	Sig[1] = 3.33;   ek[1] = 137.7;//ethylene
	Sig[2] = 2.827;  ek[2] = 59.7;//hydrogen
	Sig[3] = 3.7327; ek[3] = 149.92;//methane
	Sig[4] = 3.8;    ek[4] = 144.0;//CH3*
	Sig[5] = 4.302;  ek[5] = 252.30;//C2H5*
	Sig[6] = 2.050;  ek[6] = 145.0;//H* 

	for (int i = 1; i <= N; i++) {
		temp[i] = temp0;
		pi[i] = 0.;
		u[i] = 0.;
		for (int iMat = 0; iMat < nMat; iMat++) y_m[iMat] = 0.;
		y_m[0] = 1.;
		urs_mixt(y_m, 0.0, P0, temp[i], 2,
			tmp, gam, r, h_);
		ro[i] = r;
		rh[i] = r * h_;
		for (int iM = 0; iM < nMat; iM++) ry[iM][i] = r * y_m[iM];
	}

}

int main() {


	initializeDevice();
	mem_alloc();
	init();

	//циклы внутри шага по времени
	cl::Kernel init(program, "init");
	cl::Kernel calc_int(program, "calc_int");
	cl::Kernel calc_int1(program, "calc_int1");
	cl::Kernel calc_DTH(program, "calc_DTH");
	cl::Kernel calc_u(program, "calc_u");
	cl::Kernel calc_pi(program, "calc_pi");

	//вызываемые методы внутри шага по времени
	cl::Kernel bnd_cond(program, "bnd_cond");
	cl::Kernel calc_s(program, "calc_s");
	cl::Kernel calc_r(program, "calc_r");
	cl::Kernel calc_chem(program, "calc_chem");
	cl::Kernel calc_urs(program, "calc_urs");
	cl::Kernel calc_urs_mixt(program, "calc_urs_mixt");
	
	//выделение памяти под буфферы opencl
	cl::Buffer buffer_u(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ro(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_rh(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_pi(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_temp(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry1(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry2(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry3(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry4(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry5(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry6(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry7(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));

	cl::Buffer buffer_ro_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_u_intt(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_rh_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry1_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry2_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry3_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry4_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry5_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry6_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry7_int(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));

	cl::Buffer buffer_ro_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_u_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_rh_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_temp_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry1_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry2_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry3_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry4_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry5_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry6_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));
	cl::Buffer buffer_ry7_old(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));

	cl::Buffer buffer_S(context, CL_MEM_READ_WRITE, sizeof(double) * (N + 2));

	cl::Buffer buffer_k_TCP1(context, CL_MEM_READ_ONLY, sizeof(double) * (n_TCP));
	cl::Buffer buffer_k_TCP2(context, CL_MEM_READ_ONLY, sizeof(double) * (n_TCP));
	cl::Buffer buffer_k_TCP3(context, CL_MEM_READ_ONLY, sizeof(double) * (n_TCP));
	cl::Buffer buffer_k_TCP4(context, CL_MEM_READ_ONLY, sizeof(double) * (n_TCP));
	cl::Buffer buffer_k_TCP5(context, CL_MEM_READ_ONLY, sizeof(double) * (n_TCP));
	cl::Buffer buffer_k_TCP6(context, CL_MEM_READ_ONLY, sizeof(double) * (n_TCP));
	cl::Buffer buffer_k_TCP7(context, CL_MEM_READ_ONLY, sizeof(double) * (n_TCP));
	cl::Buffer buffer_h0(context, CL_MEM_READ_ONLY, sizeof(double) * (nMat));
	cl::Buffer buffer_KP0(context, CL_MEM_READ_ONLY, sizeof(double) * (nMat));
	cl::Buffer buffer_ML0(context, CL_MEM_READ_ONLY, sizeof(double) * (nMat));
	cl::Buffer buffer_Mm(context, CL_MEM_READ_ONLY, sizeof(double) * (nMat));
	cl::Buffer buffer_En(context, CL_MEM_READ_ONLY, sizeof(double) * (nst));
	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(double) * (nst));
	cl::Buffer buffer_hii(context, CL_MEM_READ_WRITE, sizeof(double) * (n_TCP));
	cl::Buffer buffer_Ri(context, CL_MEM_READ_WRITE, sizeof(double) * (N * nMat));
	cl::Buffer buffer_urs(context, CL_MEM_READ_WRITE, sizeof(double) * (10));

	double t = 0.;
	int step = 0;

	cl::CommandQueue queue(context, device);

	//инициализация буфферов opencl
	queue.enqueueWriteBuffer(buffer_u, CL_TRUE, 0, sizeof(double) * (N + 2), u);
	queue.enqueueWriteBuffer(buffer_ro, CL_TRUE, 0, sizeof(double) * (N + 2), ro);
	queue.enqueueWriteBuffer(buffer_rh, CL_TRUE, 0, sizeof(double) * (N + 2), rh);
	queue.enqueueWriteBuffer(buffer_pi, CL_TRUE, 0, sizeof(double) * (N + 2), pi);
	queue.enqueueWriteBuffer(buffer_temp, CL_TRUE, 0, sizeof(double) * (N + 2), temp);
	queue.enqueueWriteBuffer(buffer_ry1, CL_TRUE, 0, sizeof(double) * (N + 2), ry[0]);
	queue.enqueueWriteBuffer(buffer_ry2, CL_TRUE, 0, sizeof(double) * (N + 2), ry[1]);
	queue.enqueueWriteBuffer(buffer_ry3, CL_TRUE, 0, sizeof(double) * (N + 2), ry[2]);
	queue.enqueueWriteBuffer(buffer_ry4, CL_TRUE, 0, sizeof(double) * (N + 2), ry[3]);
	queue.enqueueWriteBuffer(buffer_ry5, CL_TRUE, 0, sizeof(double) * (N + 2), ry[4]);
	queue.enqueueWriteBuffer(buffer_ry6, CL_TRUE, 0, sizeof(double) * (N + 2), ry[5]);
	queue.enqueueWriteBuffer(buffer_ry7, CL_TRUE, 0, sizeof(double) * (N + 2), ry[6]);

	queue.enqueueWriteBuffer(buffer_ro_int, CL_TRUE, 0, sizeof(double) * (N + 2), ro_int);
	queue.enqueueWriteBuffer(buffer_u_intt, CL_TRUE, 0, sizeof(double) * (N + 2), u_intt);
	queue.enqueueWriteBuffer(buffer_rh_int, CL_TRUE, 0, sizeof(double) * (N + 2), rh_int);
	queue.enqueueWriteBuffer(buffer_ry1_int, CL_TRUE, 0, sizeof(double) * (N + 2), ry_int[0]);
	queue.enqueueWriteBuffer(buffer_ry2_int, CL_TRUE, 0, sizeof(double) * (N + 2), ry_int[1]);
	queue.enqueueWriteBuffer(buffer_ry3_int, CL_TRUE, 0, sizeof(double) * (N + 2), ry_int[2]);
	queue.enqueueWriteBuffer(buffer_ry4_int, CL_TRUE, 0, sizeof(double) * (N + 2), ry_int[3]);
	queue.enqueueWriteBuffer(buffer_ry5_int, CL_TRUE, 0, sizeof(double) * (N + 2), ry_int[4]);
	queue.enqueueWriteBuffer(buffer_ry6_int, CL_TRUE, 0, sizeof(double) * (N + 2), ry_int[5]);
	queue.enqueueWriteBuffer(buffer_ry7_int, CL_TRUE, 0, sizeof(double) * (N + 2), ry_int[6]);

	queue.enqueueWriteBuffer(buffer_ro_old, CL_TRUE, 0, sizeof(double) * (N + 2), ro_old);
	queue.enqueueWriteBuffer(buffer_u_old, CL_TRUE, 0, sizeof(double) * (N + 2), u_old);
	queue.enqueueWriteBuffer(buffer_rh_old, CL_TRUE, 0, sizeof(double) * (N + 2), rh_old);
	queue.enqueueWriteBuffer(buffer_temp_old, CL_TRUE, 0, sizeof(double) * (N + 2), temp_old);
	queue.enqueueWriteBuffer(buffer_ry1_old, CL_TRUE, 0, sizeof(double) * (N + 2), ry_old[0]);
	queue.enqueueWriteBuffer(buffer_ry2_old, CL_TRUE, 0, sizeof(double) * (N + 2), ry_old[1]);
	queue.enqueueWriteBuffer(buffer_ry3_old, CL_TRUE, 0, sizeof(double) * (N + 2), ry_old[2]);
	queue.enqueueWriteBuffer(buffer_ry4_old, CL_TRUE, 0, sizeof(double) * (N + 2), ry_old[3]);
	queue.enqueueWriteBuffer(buffer_ry5_old, CL_TRUE, 0, sizeof(double) * (N + 2), ry_old[4]);
	queue.enqueueWriteBuffer(buffer_ry6_old, CL_TRUE, 0, sizeof(double) * (N + 2), ry_old[5]);
	queue.enqueueWriteBuffer(buffer_ry7_old, CL_TRUE, 0, sizeof(double) * (N + 2), ry_old[6]);

	queue.enqueueWriteBuffer(buffer_S, CL_TRUE, 0, sizeof(double) * (N + 2), S);

	queue.enqueueWriteBuffer(buffer_k_TCP1, CL_TRUE, 0, sizeof(double) * (n_TCP), k_TCP[0]);
	queue.enqueueWriteBuffer(buffer_k_TCP2, CL_TRUE, 0, sizeof(double) * (n_TCP), k_TCP[1]);
	queue.enqueueWriteBuffer(buffer_k_TCP3, CL_TRUE, 0, sizeof(double) * (n_TCP), k_TCP[2]);
	queue.enqueueWriteBuffer(buffer_k_TCP4, CL_TRUE, 0, sizeof(double) * (n_TCP), k_TCP[3]);
	queue.enqueueWriteBuffer(buffer_k_TCP5, CL_TRUE, 0, sizeof(double) * (n_TCP), k_TCP[4]);
	queue.enqueueWriteBuffer(buffer_k_TCP6, CL_TRUE, 0, sizeof(double) * (n_TCP), k_TCP[5]);
	queue.enqueueWriteBuffer(buffer_k_TCP7, CL_TRUE, 0, sizeof(double) * (n_TCP), k_TCP[6]);
	queue.enqueueWriteBuffer(buffer_h0, CL_TRUE, 0, sizeof(double) * (nMat), h0);
	queue.enqueueWriteBuffer(buffer_KP0, CL_TRUE, 0, sizeof(double) * (n_TCP), KP0);
	queue.enqueueWriteBuffer(buffer_ML0, CL_TRUE, 0, sizeof(double) * (n_TCP), ML0);
	queue.enqueueWriteBuffer(buffer_hii, CL_TRUE, 0, sizeof(double) * (n_TCP), hii);
	queue.enqueueWriteBuffer(buffer_Mm, CL_TRUE, 0, sizeof(double) * (nMat), Mm);
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(double) * (nst), A);
	queue.enqueueWriteBuffer(buffer_En, CL_TRUE, 0, sizeof(double) * (nst), En);
	queue.enqueueWriteBuffer(buffer_Ri, CL_TRUE, 0, sizeof(double) * (N * nMat), Ri);
	queue.enqueueWriteBuffer(buffer_urs, CL_TRUE, 0, sizeof(double) * (10), urs);

	int ix = 0;
	init.setArg(ix++, buffer_ry1_int);
	init.setArg(ix++, buffer_ry2_int);
	init.setArg(ix++, buffer_ry3_int);
	init.setArg(ix++, buffer_ry4_int);
	init.setArg(ix++, buffer_ry5_int);
	init.setArg(ix++, buffer_ry6_int);
	init.setArg(ix++, buffer_ry7_int);
	init.setArg(ix++, buffer_ry1_old);
	init.setArg(ix++, buffer_ry2_old);
	init.setArg(ix++, buffer_ry3_old);
	init.setArg(ix++, buffer_ry4_old);
	init.setArg(ix++, buffer_ry5_old);
	init.setArg(ix++, buffer_ry6_old);
	init.setArg(ix++, buffer_ry7_old);
	init.setArg(ix++, buffer_ry1);
	init.setArg(ix++, buffer_ry2);
	init.setArg(ix++, buffer_ry3);
	init.setArg(ix++, buffer_ry4);
	init.setArg(ix++, buffer_ry5);
	init.setArg(ix++, buffer_ry6);
	init.setArg(ix++, buffer_ry7);
	init.setArg(ix++, buffer_ro_int);
	init.setArg(ix++, buffer_u_intt);
	init.setArg(ix++, buffer_rh_int);
	init.setArg(ix++, buffer_S);
	init.setArg(ix++, N);

	ix = 0;
	calc_chem.setArg(ix++, buffer_ry1_old);
	calc_chem.setArg(ix++, buffer_ry2_old);
	calc_chem.setArg(ix++, buffer_ry3_old);
	calc_chem.setArg(ix++, buffer_ry4_old);
	calc_chem.setArg(ix++, buffer_ry5_old);
	calc_chem.setArg(ix++, buffer_ry6_old);
	calc_chem.setArg(ix++, buffer_ry7_old);
	calc_chem.setArg(ix++, buffer_temp);
	calc_chem.setArg(ix++, buffer_Mm);
	calc_chem.setArg(ix++, buffer_A);
	calc_chem.setArg(ix++, buffer_En);
	calc_chem.setArg(ix++, N);
	calc_chem.setArg(ix++, gR);
	calc_chem.setArg(ix++, TAUH);
	calc_chem.setArg(ix++, nst);
	calc_chem.setArg(ix++, nMat);

	ix = 0;
	calc_int.setArg(ix++, buffer_u);
	calc_int.setArg(ix++, buffer_ro);
	calc_int.setArg(ix++, buffer_rh);
	calc_int.setArg(ix++, buffer_pi);
	calc_int.setArg(ix++, buffer_temp);
	calc_int.setArg(ix++, buffer_u_intt);
	calc_int.setArg(ix++, buffer_rh_int);
	calc_int.setArg(ix++, buffer_ry1);
	calc_int.setArg(ix++, buffer_ry2);
	calc_int.setArg(ix++, buffer_ry3);
	calc_int.setArg(ix++, buffer_ry4);
	calc_int.setArg(ix++, buffer_ry5);
	calc_int.setArg(ix++, buffer_ry6);
	calc_int.setArg(ix++, buffer_ry7);
	calc_int.setArg(ix++, buffer_ry1_int);
	calc_int.setArg(ix++, buffer_ry2_int);
	calc_int.setArg(ix++, buffer_ry3_int);
	calc_int.setArg(ix++, buffer_ry4_int);
	calc_int.setArg(ix++, buffer_ry5_int);
	calc_int.setArg(ix++, buffer_ry6_int);
	calc_int.setArg(ix++, buffer_ry7_int);
	calc_int.setArg(ix++, buffer_k_TCP1);
	calc_int.setArg(ix++, buffer_k_TCP2);
	calc_int.setArg(ix++, buffer_k_TCP3);
	calc_int.setArg(ix++, buffer_k_TCP4);
	calc_int.setArg(ix++, buffer_k_TCP5);
	calc_int.setArg(ix++, buffer_k_TCP6);
	calc_int.setArg(ix++, buffer_k_TCP7);
	calc_int.setArg(ix++, buffer_Mm);
	calc_int.setArg(ix++, buffer_KP0);
	calc_int.setArg(ix++, buffer_ML0);
	calc_int.setArg(ix++, buffer_h0);
	calc_int.setArg(ix++, N);
	calc_int.setArg(ix++, OsnV);
	calc_int.setArg(ix++, nMat);
	calc_int.setArg(ix++, n_TCP);
	calc_int.setArg(ix++, H);
	calc_int.setArg(ix++, P0);
	ix = 0;
	calc_int1.setArg(ix++, buffer_u);
	calc_int1.setArg(ix++, buffer_ro);
	calc_int1.setArg(ix++, buffer_rh);
	calc_int1.setArg(ix++, buffer_pi);
	calc_int1.setArg(ix++, buffer_temp);
	calc_int1.setArg(ix++, buffer_u_intt);
	calc_int1.setArg(ix++, buffer_rh_int);
	calc_int1.setArg(ix++, buffer_ry1);
	calc_int1.setArg(ix++, buffer_ry2);
	calc_int1.setArg(ix++, buffer_ry3);
	calc_int1.setArg(ix++, buffer_ry4);
	calc_int1.setArg(ix++, buffer_ry5);
	calc_int1.setArg(ix++, buffer_ry6);
	calc_int1.setArg(ix++, buffer_ry7);
	calc_int1.setArg(ix++, buffer_ry1_int);
	calc_int1.setArg(ix++, buffer_ry2_int);
	calc_int1.setArg(ix++, buffer_ry3_int);
	calc_int1.setArg(ix++, buffer_ry4_int);
	calc_int1.setArg(ix++, buffer_ry5_int);
	calc_int1.setArg(ix++, buffer_ry6_int);
	calc_int1.setArg(ix++, buffer_ry7_int);
	calc_int1.setArg(ix++, buffer_k_TCP1);
	calc_int1.setArg(ix++, buffer_k_TCP2);
	calc_int1.setArg(ix++, buffer_k_TCP3);
	calc_int1.setArg(ix++, buffer_k_TCP4);
	calc_int1.setArg(ix++, buffer_k_TCP5);
	calc_int1.setArg(ix++, buffer_k_TCP6);
	calc_int1.setArg(ix++, buffer_k_TCP7);
	calc_int1.setArg(ix++, buffer_Mm);
	calc_int1.setArg(ix++, buffer_KP0);
	calc_int1.setArg(ix++, buffer_ML0);
	calc_int1.setArg(ix++, buffer_h0);
	calc_int1.setArg(ix++, N);
	calc_int1.setArg(ix++, OsnV);
	calc_int1.setArg(ix++, nMat);
	calc_int1.setArg(ix++, n_TCP);
	calc_int1.setArg(ix++, H);
	calc_int1.setArg(ix++, P0);

	ix = 0;
	calc_DTH.setArg(ix++, buffer_u);
	calc_DTH.setArg(ix++, buffer_ro);
	calc_DTH.setArg(ix++, buffer_rh);
	calc_DTH.setArg(ix++, buffer_u_intt);
	calc_DTH.setArg(ix++, buffer_rh_int);
	calc_DTH.setArg(ix++, buffer_ry1);
	calc_DTH.setArg(ix++, buffer_ry2);
	calc_DTH.setArg(ix++, buffer_ry3);
	calc_DTH.setArg(ix++, buffer_ry4);
	calc_DTH.setArg(ix++, buffer_ry5);
	calc_DTH.setArg(ix++, buffer_ry6);
	calc_DTH.setArg(ix++, buffer_ry7);
	calc_DTH.setArg(ix++, buffer_ry1_old);
	calc_DTH.setArg(ix++, buffer_ry2_old);
	calc_DTH.setArg(ix++, buffer_ry3_old);
	calc_DTH.setArg(ix++, buffer_ry4_old);
	calc_DTH.setArg(ix++, buffer_ry5_old);
	calc_DTH.setArg(ix++, buffer_ry6_old);
	calc_DTH.setArg(ix++, buffer_ry7_old);
	calc_DTH.setArg(ix++, buffer_ry1_int);
	calc_DTH.setArg(ix++, buffer_ry2_int);
	calc_DTH.setArg(ix++, buffer_ry3_int);
	calc_DTH.setArg(ix++, buffer_ry4_int);
	calc_DTH.setArg(ix++, buffer_ry5_int);
	calc_DTH.setArg(ix++, buffer_ry6_int);
	calc_DTH.setArg(ix++, buffer_ry7_int);
	calc_DTH.setArg(ix++, N);
	calc_DTH.setArg(ix++, DTH);

	ix = 0;
	calc_u.setArg(ix++, buffer_u);
	calc_u.setArg(ix++, buffer_ro);
	calc_u.setArg(ix++, buffer_pi);
	calc_u.setArg(ix++, N);
	calc_u.setArg(ix++, DT);
	calc_u.setArg(ix++, H);

	ix = 0;
	calc_s.setArg(ix++, buffer_S);
	calc_s.setArg(ix++, buffer_Mm);
	calc_s.setArg(ix++, buffer_ry1);
	calc_s.setArg(ix++, buffer_ry2);
	calc_s.setArg(ix++, buffer_ry3);
	calc_s.setArg(ix++, buffer_ry4);
	calc_s.setArg(ix++, buffer_ry5);
	calc_s.setArg(ix++, buffer_ry6);
	calc_s.setArg(ix++, buffer_ry7);
	calc_s.setArg(ix++, buffer_ro);
	calc_s.setArg(ix++, buffer_pi);
	calc_s.setArg(ix++, buffer_temp);
	calc_s.setArg(ix++, buffer_k_TCP1);
	calc_s.setArg(ix++, buffer_k_TCP2);
	calc_s.setArg(ix++, buffer_k_TCP3);
	calc_s.setArg(ix++, buffer_k_TCP4);
	calc_s.setArg(ix++, buffer_k_TCP5);
	calc_s.setArg(ix++, buffer_k_TCP6);
	calc_s.setArg(ix++, buffer_k_TCP7);
	calc_s.setArg(ix++, buffer_KP0);
	calc_s.setArg(ix++, buffer_h0);
	calc_s.setArg(ix++, buffer_Ri);
	calc_s.setArg(ix++, N);
	calc_s.setArg(ix++, OsnV);
	calc_s.setArg(ix++, nMat);
	calc_s.setArg(ix++, n_TCP);
	calc_s.setArg(ix++, DT);
	calc_s.setArg(ix++, H);
	calc_s.setArg(ix++, P0);

	ix = 0;
	calc_urs.setArg(ix++, buffer_Mm);
	calc_urs.setArg(ix++, buffer_hii);
	calc_urs.setArg(ix++, buffer_urs);
	calc_urs.setArg(ix++, buffer_k_TCP1);
	calc_urs.setArg(ix++, buffer_k_TCP2);
	calc_urs.setArg(ix++, buffer_k_TCP3);
	calc_urs.setArg(ix++, buffer_k_TCP4);
	calc_urs.setArg(ix++, buffer_k_TCP5);
	calc_urs.setArg(ix++, buffer_k_TCP6);
	calc_urs.setArg(ix++, buffer_k_TCP7);
	calc_urs.setArg(ix++, buffer_h0);
	calc_urs.setArg(ix++, 0.0);
	calc_urs.setArg(ix++, temp_in);
	calc_urs.setArg(ix++, P0);
	calc_urs.setArg(ix++, gR);
	calc_urs.setArg(ix++, nMat);
	calc_urs.setArg(ix++, n_TCP);

	ix = 0;
	bnd_cond.setArg(ix++, buffer_u);
	bnd_cond.setArg(ix++, buffer_ro);
	bnd_cond.setArg(ix++, buffer_rh);
	bnd_cond.setArg(ix++, buffer_pi);
	bnd_cond.setArg(ix++, buffer_temp);
	bnd_cond.setArg(ix++, buffer_ry1);
	bnd_cond.setArg(ix++, buffer_ry2);
	bnd_cond.setArg(ix++, buffer_ry3);
	bnd_cond.setArg(ix++, buffer_ry4);
	bnd_cond.setArg(ix++, buffer_ry5);
	bnd_cond.setArg(ix++, buffer_ry6);
	bnd_cond.setArg(ix++, buffer_ry7);
	bnd_cond.setArg(ix++, buffer_urs);
	bnd_cond.setArg(ix++, N);
	bnd_cond.setArg(ix++, nMat);
	bnd_cond.setArg(ix++, temp_in);
	bnd_cond.setArg(ix++, P0);
	bnd_cond.setArg(ix++, u_g);

	ix = 0;
	calc_urs_mixt.setArg(ix++, buffer_k_TCP1);
	calc_urs_mixt.setArg(ix++, buffer_k_TCP2);
	calc_urs_mixt.setArg(ix++, buffer_k_TCP3);
	calc_urs_mixt.setArg(ix++, buffer_k_TCP4);
	calc_urs_mixt.setArg(ix++, buffer_k_TCP5);
	calc_urs_mixt.setArg(ix++, buffer_k_TCP6);
	calc_urs_mixt.setArg(ix++, buffer_k_TCP7);
	calc_urs_mixt.setArg(ix++, buffer_ry1);
	calc_urs_mixt.setArg(ix++, buffer_ry2);
	calc_urs_mixt.setArg(ix++, buffer_ry3);
	calc_urs_mixt.setArg(ix++, buffer_ry4);
	calc_urs_mixt.setArg(ix++, buffer_ry5);
	calc_urs_mixt.setArg(ix++, buffer_ry6);
	calc_urs_mixt.setArg(ix++, buffer_ry7);
	calc_urs_mixt.setArg(ix++, buffer_h0);
	calc_urs_mixt.setArg(ix++, buffer_rh);
	calc_urs_mixt.setArg(ix++, buffer_ro);
	calc_urs_mixt.setArg(ix++, buffer_temp);
	calc_urs_mixt.setArg(ix++, P0);
	calc_urs_mixt.setArg(ix++, gR);
	calc_urs_mixt.setArg(ix++, nMat);
	calc_urs_mixt.setArg(ix++, n_TCP);
	calc_urs_mixt.setArg(ix++, N);

	ix = 0;
	calc_pi.setArg(ix++, buffer_u);
	calc_pi.setArg(ix++, buffer_ro);
	calc_pi.setArg(ix++, buffer_pi);
	calc_pi.setArg(ix++, buffer_S);
	calc_pi.setArg(ix++, N);
	calc_pi.setArg(ix++, DT);
	calc_pi.setArg(ix++, H);

	ix = 0;
	calc_r.setArg(ix++, buffer_Ri);
	calc_r.setArg(ix++, buffer_Mm);
	calc_r.setArg(ix++, buffer_ry1);
	calc_r.setArg(ix++, buffer_ry2);
	calc_r.setArg(ix++, buffer_ry3);
	calc_r.setArg(ix++, buffer_ry4);
	calc_r.setArg(ix++, buffer_ry5);
	calc_r.setArg(ix++, buffer_ry6);
	calc_r.setArg(ix++, buffer_ry7);
	calc_r.setArg(ix++, buffer_temp);
	calc_r.setArg(ix++, buffer_A);
	calc_r.setArg(ix++, buffer_En);
	calc_r.setArg(ix++, N);
	calc_r.setArg(ix++, nMat);
	calc_r.setArg(ix++, nst);
	calc_r.setArg(ix++, gR);
	//urs[0] = tmp, urs[1] = gamma, urs[2] = r, urs[3] = h_, urs[4] = fM, urs[5] = tmp1, urs[6] = Cv, urs[7] = Cp
	//queue.enqueueWriteBuffer(buffer_h0, CL_TRUE, 0, sizeof(double)* (n_TCP), h0);
	//for (int i = 0; i < nMat; i++) {
	//	printf("%f ", h0[i]);
	//}
	double start_time = clock(); // начальное время
	while ((t < TMAX) && (step != save_step)) {

		t += DT; step++;

		queue.enqueueNDRangeKernel(calc_urs, cl::NullRange, cl::NDRange(1), cl::NullRange);

		queue.enqueueNDRangeKernel(bnd_cond, cl::NullRange, cl::NDRange(1), cl::NullRange);
		
		queue.enqueueNDRangeKernel(init, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);
		
		for (int k = 0; k < nsteph - 1; k++) {
			queue.enqueueNDRangeKernel(calc_chem, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);
		}
		
		queue.enqueueNDRangeKernel(calc_int, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);
		queue.enqueueNDRangeKernel(calc_int1, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);
		
		queue.enqueueNDRangeKernel(calc_DTH, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);

		queue.enqueueNDRangeKernel(calc_urs_mixt, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);

		queue.enqueueNDRangeKernel(calc_urs, cl::NullRange, cl::NDRange(1), cl::NullRange);
		
		queue.enqueueNDRangeKernel(bnd_cond, cl::NullRange, cl::NDRange(1), cl::NullRange);
		
		queue.enqueueNDRangeKernel(calc_r, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);

		queue.enqueueNDRangeKernel(calc_s, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);
		//queue.enqueueReadBuffer(buffer_S, CL_TRUE, 0, sizeof(double) * (N + 2), S);
		//for (int i = 1; i <= N; i++) {
		//	printf("%f ", S[i]);
		//}

		queue.enqueueNDRangeKernel(calc_pi, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);

		queue.enqueueNDRangeKernel(calc_u, cl::NullRange, cl::NDRange(N + 2), cl::NullRange);

		std::cout << "STEP: " << step << std::endl;


		if (step % save_step == 0) {
			queue.enqueueReadBuffer(buffer_u, CL_TRUE, 0, sizeof(double) * (N + 2), u);
			queue.enqueueReadBuffer(buffer_ro, CL_TRUE, 0, sizeof(double) * (N + 2), ro);
			queue.enqueueReadBuffer(buffer_pi, CL_TRUE, 0, sizeof(double) * (N + 2), pi);
			queue.enqueueReadBuffer(buffer_temp, CL_TRUE, 0, sizeof(double) * (N + 2), temp);
			queue.enqueueReadBuffer(buffer_ry1, CL_TRUE, 0, sizeof(double) * (N + 2), ry[0]);
			queue.enqueueReadBuffer(buffer_ry2, CL_TRUE, 0, sizeof(double) * (N + 2), ry[1]);
			queue.enqueueReadBuffer(buffer_ry3, CL_TRUE, 0, sizeof(double) * (N + 2), ry[2]);
			queue.enqueueReadBuffer(buffer_ry4, CL_TRUE, 0, sizeof(double) * (N + 2), ry[3]);
			queue.finish();
			//for (int i = 1; i <= N; i++) {
			//	printf("%f ", temp[i]);
			//}
			FILE* fp;
			char fName[50]; 

			sprintf(fName, "resOpenCL_%d_%d.csv", step, N);
			fp = fopen(fName, "w");

			fprintf(fp, "x,c2h6,c2h4,h2,ch4,ro,u,temp,pi \n");
			for (int i = 1; i <= N; i++) {
				fprintf(fp, "%25.16f, ", (i - 1) * H);
				fprintf(fp, "%25.16f, ", ry[0][i] / ro[i]);
				fprintf(fp, "%25.16f, ", ry[1][i] / ro[i]);
				fprintf(fp, "%25.16f, ", ry[2][i] / ro[i]);
				fprintf(fp, "%25.16f, ", ry[3][i] / ro[i]);
				fprintf(fp, "%25.16f, ", ro[i]);
				fprintf(fp, "%25.16f, ", u[i]);
				fprintf(fp, "%25.16f, ", temp[i]);
				fprintf(fp, "%25.16f ", pi[i]);
				fprintf(fp, "\n");
			}
			fprintf(fp, "\n");
			fclose(fp);
		}
	}

	std::cout << "time: " << (clock() - start_time) / CLOCKS_PER_SEC << std::endl;
	mem_free();
	return 0;
}

//уравнение состояния для многокомпонентного идеального газа
void urs_mixt(double* y_m, double h, double p, double fT, int flag,
	double& tmp, double& gamma, double& r, double& h_)
{
	double Cv, Cp, e;
	double fM = 0.0; // fM = SUM( Yi / Mi )
	tmp = fT;
	double tmp1 = fT;
	for (int iM = 0; iM < nMat; iM++) fM += y_m[iM] / Mm[iM];

	// вычисляем Т итерациями по Ньютону
	if (flag == 1) {
		Newton_Method(h, y_m, tmp1,
			tmp);
	}

	double fCP = 0.0;
	for (int iM = 0; iM < nMat; iM++)  fCP += y_m[iM] * Calc_CP(iM, tmp);

	Cp = fCP;
	Cv = Cp - gR * fM;
	gamma = Cp / Cv;

	switch (flag)
	{
	case 2:		// r=r(T,p), h = sum_h(i)
		calc_h(tmp,
			hii);
		h_ = 0.0;
		for (int iM = 0; iM < nMat; iM++) {
			h_ += y_m[iM] * hii[iM];
		}
		r = P0 / (gR * tmp * fM);
		break;
	}
}

void calc_h(double tmp,
	double hii[nMat])
{
	double TRef = 298.15;
	double t1, t2;
	for (int iM = 0; iM < nMat; iM++) {
		hii[iM] = h0[iM];
		for (int i = 0; i < n_TCP; i++) {
			int q = i + 1;
			t2 = exp((q)*log(tmp)) / (q);
			t1 = exp((q)*log(TRef)) / (q);
			hii[iM] += k_TCP[iM][i] * (t2 - t1);
		}
	}
}

double Calc_CP(int iM, double fT)
{
	double fTP = fT, fCP = k_TCP[iM][0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP[iM][i];
		fTP *= fT;
	}
	return fCP;
}

double Calc_CP_(int iM, double fT)
{
	int n_TCP_ = n_TCP - 1;
	double* k_TCP_ = new double[n_TCP_];
	for (int i = 0; i < n_TCP_; i++) k_TCP_[i] = (i + 1) * k_TCP[iM][i + 1];

	double fTP = fT, fCP = k_TCP_[0];
	for (int i = 1; i < n_TCP_; i++) {
		fCP += fTP * k_TCP_[i];
		fTP *= fT;
	}
	delete[] k_TCP_;
	return fCP;
}

void Newton_Method(double e, double* y_m, double tmp1,
	double& tmp)
{
	double fT = tmp1;		// начальное приближение для температуры
	double fE = e;		// энтальпия

	double fM = 0.0; // fM = SUM( Yi / Mi )
	for (int i = 0; i < nMat; i++) fM += y_m[i] / Mm[i];
	double fR_M = gR * fM;
	fR_M = 0.0;

	for (int ic = 0; ic < 100; ic++)
	{
		double fCP = 0.0;
		double fCP_ = 0.0;
		calc_h(fT,
			hii);
		for (int i = 0; i < nMat; i++)
		{
			fCP += y_m[i] * hii[i];
			fCP_ += y_m[i] * Calc_CP(i, fT);
		}

		double fFT = fCP - fE;
		double fFT_ = fCP_;

		double fTg = fT - fFT / fFT_;
		if ((fTg - fT) * (fTg - fT) < 1.0e-10) { fT = fTg; break; }
		fT = fTg;
	}
	tmp = fT;
}

void mem_alloc()
{
	ro = new double[N + 2];
	u = new double[N + 2];
	rh = new double[N + 2];
	pi = new double[N + 2];
	temp = new double[N + 2];
	ry = new double* [nMat];
	urs = new double[10];
	for (int i = 0; i < nMat; i++) ry[i] = new double[N + 2];

	ro_int = new double[N + 2];
	u_intt = new double[N + 2];
	rh_int = new double[N + 2];
	ry_int = new double* [nMat];
	for (int i = 0; i < nMat; i++) ry_int[i] = new double[N + 2];
	S = new double[N + 2];

	ro_old = new double[N + 2];
	u_old = new double[N + 2];
	rh_old = new double[N + 2];
	temp_old = new double[N + 2];
	ry_old = new double* [nMat];
	for (int i = 0; i < nMat; i++) ry_old[i] = new double[N + 2];

	Mm = new double[nMat];
	h0 = new double[nMat];

	ML0 = new double[nMat];
	y_m = new double[nMat];
	KP0 = new double[nMat];
	Sig = new double[nMat];
	hii = new double[nMat];
	ek = new double[nMat];

	Ri = new double [N * nMat];

	DF = new double[nMat];
	DFr = new double[nMat];
	DFl = new double[nMat];
}


void mem_free()
{
	delete[] ro;
	delete[] u;
	delete[] rh;
	delete[] pi;
	delete[] temp;
	for (int i = 0; i < nMat; i++) {
		delete[] ry[i];
	}
	delete[] ry;
	delete[] urs;
	delete[] ro_int;
	delete[] u_intt;
	delete[] rh_int;
	for (int i = 0; i < nMat; i++) {
		delete[] ry_int[i];
	}
	delete[] ry_int;
	delete[] S;

	delete[] ro_old;
	delete[] u_old;
	delete[] rh_old;
	delete[] temp_old;
	for (int i = 0; i < nMat; i++) {
		delete[] ry_old[i];
	}
	delete[] ry_old;

	delete[] y;
	delete[] y_m;
	delete[] yl;
	delete[] yr;
	delete[] fry;

	delete[] Mm;
	delete[] h0;


	delete[] hii;
	delete[] hir;
	delete[] hil;
}


