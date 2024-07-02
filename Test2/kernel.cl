double Calc_CP(double* y, double fT, double* k_TCP1, double* k_TCP2, double* k_TCP3, double* k_TCP4, double* k_TCP5, double* k_TCP6, double* k_TCP7)
{
	double fTP = fT, fCP = k_TCP1[0], res = 0;
	const int n_TCP = 4;
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP1[i];
		fTP *= fT;
	}
	res += y[0] * fCP;

	fTP = fT, fCP = k_TCP2[0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP2[i];
		fTP *= fT;
	}
	res += y[1] * fCP;

	fTP = fT, fCP = k_TCP3[0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP3[i];
		fTP *= fT;
	}
	res += y[2] * fCP;

	fTP = fT, fCP = k_TCP4[0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP4[i];
		fTP *= fT;
	}
	res += y[3] * fCP;

	fTP = fT, fCP = k_TCP5[0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP5[i];
		fTP *= fT;
	}
	res += y[4] * fCP;

	fTP = fT, fCP = k_TCP6[0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP6[i];
		fTP *= fT;
	}
	res += y[5] * fCP;

	fTP = fT, fCP = k_TCP7[0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP7[i];
		fTP *= fT;
	}
	res += y[6] * fCP;
	return res;
}
double Calc_KP_Nv(double* y_m, double* Mm, double tmp, int nMat, int OsnV,  double* KP0)
{
	double fKP = 0.0;
	double fM_ = 0.0;
	double KP = 0.0;
	double KPm[7];

	KPm[0] = (1.11e-7) * tmp * tmp + (6.13e-5) * tmp + (-5.767e-3);
	KPm[1] = (1.07e-7) * tmp * tmp + (5.41e-5) * tmp + (-4.72e-3);
	KPm[2] = (1.07e-8) * tmp * tmp + (0.00036) * tmp + (0.0827);
	KPm[3] = (0.0) * tmp * tmp + (0.0002) * tmp + (-0.0272);


	for (int iM = 0; iM < nMat; iM++)
	{
		double fCK_M = y_m[iM] / Mm[iM];
		fM_ += fCK_M;
		if (iM < OsnV) {
			fKP += fCK_M * KPm[iM];
		}
		else {
			fKP += fCK_M * KP0[iM];
		}
	}

	KP = fKP / fM_;

	return KP;
}
void calc_Dm(double* y_m, double p, double tmp, double* Dm, double* Mm, int nMat)
{
	double pabs = p / 101325.0;
	double ek[7];
	double Sig[7];
	double Dij[7 * 7];

	ek[0] = 139.8;  Sig[0] = 3.512;
	ek[1] = 137.7;  Sig[1] = 3.33;
	ek[2] = 59.7;   Sig[2] = 2.827;
	ek[3] = 149.92; Sig[3] = 3.7327;
	ek[4] = 144.0;  Sig[4] = 3.8;
	ek[5] = 252.30; Sig[5] = 4.302;
	ek[6] = 145.0;  Sig[6] = 2.050;

	int k = 0;
	for (int i = 0; i < nMat; i++) {
		for (int j = 0; j < nMat; j++) {
			double valTD = tmp / sqrt(ek[i] * ek[j]);
			double valWD = 1.06036 / pow(valTD, 0.1561) + 0.1930 / exp(0.47635 * valTD) + 1.03587 / exp(1.52996 * valTD) + 1.76474 / exp(3.89411 * valTD);
			Dij[k++] = (2.628e-7) * sqrt(0.5 * (1.0 / (1000.0 * Mm[i]) + 1.0 / (1000.0 * Mm[j]))) * sqrt(tmp * tmp * tmp) / (pabs * pow((0.5 * (Sig[i] + Sig[j])), 2.0) * valWD);
		}
	}

	double fM_ = 0.;
	double xk[7];

	for (int iM = 0; iM < nMat; iM++)
	{
		xk[iM] = y_m[iM] / Mm[iM];
		fM_ += xk[iM];
	}
	for (int iM = 0; iM < nMat; iM++)
	{
		xk[iM] = (y_m[iM] / Mm[iM]) / fM_;
		
	}
	for (int iM = 0; iM < nMat; iM++) {
		if (fabs(y_m[iM] - 1.) <= 1.e-20) { Dm[iM] = 0.; }
		else {
			double SumXD = 0.0;
			for (int i = 0; i < nMat; i++) {
				if (i != iM) SumXD += xk[i] / Dij[iM * nMat + i];
			}
			Dm[iM] = (1. - xk[iM]) / SumXD;
		}
	}
}
void calc_h(double* h0, double tmp, double* hii,  double* k_TCP1,  double* k_TCP2,  double* k_TCP3,  double* k_TCP4, double* k_TCP5, double* k_TCP6, double* k_TCP7, int nMat, int n_TCP)
{
	double TRef = 298.15;
	double t1, t2;
	for (int i = 0; i < nMat; i++)	hii[i] = h0[i];

	for (int i = 0; i < n_TCP; i++) {
		int q = i + 1;
		t2 = exp((q)*log(tmp)) / (q);
		t1 = exp((q)*log(TRef)) / (q);
		hii[0] += k_TCP1[i] * (t2 - t1);
		hii[1] += k_TCP2[i] * (t2 - t1);
		hii[2] += k_TCP3[i] * (t2 - t1);
		hii[3] += k_TCP4[i] * (t2 - t1);
		hii[4] += k_TCP5[i] * (t2 - t1);
		hii[5] += k_TCP6[i] * (t2 - t1);
		hii[6] += k_TCP7[i] * (t2 - t1);
	}
}
double Calc_ML_Nv(double* y_m, double* Mm, double tmp, int nMat, int OsnV,  double* ML0)
{
	double fKP = 0.0;
	double fM_ = 0.0;
	double KP = 0.0;
	double KPm[7];

	KPm[0] = (-9.02e-12) * tmp * tmp + (3.41e-8) * tmp + (9.81e-9);
	KPm[1] = (-9.45e-12) * tmp * tmp + (3.67e-8) * tmp + (0.22e-6);
	KPm[2] = (-3.23e-12) * tmp * tmp + (1.93e-8) * tmp + (3.77e-6);
	KPm[3] = (-9.85e-12) * tmp * tmp + (3.61e-8) * tmp + (1.31e-6);

	for (int iM = 0; iM < nMat; iM++)
	{
		double fCK_M = y_m[iM] / Mm[iM];
		fM_ += fCK_M;
		if (iM < OsnV) {
			fKP += fCK_M * KPm[iM];
		}
		else {
			fKP += fCK_M * ML0[iM];
		}
	}

	KP = fKP / fM_;

	return KP;
}
void calc_phi(double tmp, double* uu_,double* phi, double* psi, int nst, double* Mm, double gR, int nMat, double* A, double* En)
{

	double k[5];

	for (int ki = 0; ki < nst; ki++) { k[ki] = A[ki] * exp(-En[ki] / (gR * tmp)); }

	for (int im = 0; im < nMat; im++) uu_[im] = uu_[im] / Mm[im];

	phi[0] = k[0] + k[1] * uu_[4] + k[3] * uu_[6];
	psi[0] = Mm[0] * (k[4] * uu_[5] * uu_[5]);
	phi[1] = 0.0;
	psi[1] = Mm[1] * (k[2] * uu_[5] + k[4] * uu_[5] * uu_[5]);
	phi[2] = 0.0;
	psi[2] = Mm[2] * (k[3] * uu_[0] * uu_[6]);
	phi[3] = 0.0;
	psi[3] = Mm[3] * (k[1] * uu_[4] * uu_[0]);
	phi[4] = k[1] * uu_[0];
	psi[4] = Mm[4] * (2 * k[0] * uu_[0]);
	phi[5] = k[2] + 2 * k[4] * uu_[5];
	psi[5] = Mm[5] * (k[1] * uu_[0] * uu_[4] + k[3] * uu_[0] * uu_[6]);
	phi[6] = k[3] * uu_[0];
	psi[6] = Mm[6] * (k[2] * uu_[5]);

}
void calc_conc(double tmp, double* uu, double* uu_s, const int nMat, const double gR, const int nst, double* Mm, const double TAUH, double* A, double* En)
{

	double phi[7];
	double psi[7];

	double uu_[7];

	for (int im = 0; im < nMat; im++)
	{
		uu_s[im] = uu[im];
	}
	for (int i = 0; i < 2; i++) {
		for (int im = 0; im < nMat; im++) uu_[im] = (uu[im] + uu_s[im]) / 2.0;

		calc_phi(tmp, uu_, phi, psi, nst, Mm, gR, nMat, A, En);

		for (int im = 0; im < nMat; im++) uu_s[im] = (uu[im] + TAUH * psi[im] * (1.0 + TAUH * phi[im] / 2.0)) / (1.0 + TAUH * phi[im] + TAUH * phi[im] * TAUH * phi[im] / 2.0);
	}

}
__kernel void calc_chem(__global double* ry1_old,
	__global double* ry2_old,
	__global double* ry3_old,
	__global double* ry4_old,
	__global double* ry5_old,
	__global double* ry6_old,
	__global double* ry7_old,
	__global double* temp,
	__global double* Mm,
	__global double* A,
	__global double* En,
	const int N,
	const double gR,
	const double TAUH,
	const int nst,
	const int nMat) {
	int i = get_global_id(0);

	if ((i >= 1) && (i <= N)) {
		double uu[7];
		double uu_s[7];
		uu[0] = ry1_old[i];
		uu[1] = ry2_old[i];
		uu[2] = ry3_old[i];
		uu[3] = ry4_old[i];
		uu[4] = ry5_old[i];
		uu[5] = ry6_old[i];
		uu[6] = ry7_old[i];

		calc_conc(temp[i], uu, uu_s, nMat, gR, nst, Mm, TAUH, A, En);
		
		ry1_old[i] = uu_s[0];
		ry2_old[i] = uu_s[1];
		ry3_old[i] = uu_s[2];
		ry4_old[i] = uu_s[3];
		ry5_old[i] = uu_s[4];
		ry6_old[i] = uu_s[5];
		ry7_old[i] = uu_s[6];
		
	}
}
__kernel void init(__global double* ry1_int,
	__global double* ry2_int,
	__global double* ry3_int,
	__global double* ry4_int,
	__global double* ry5_int,
	__global double* ry6_int,
	__global double* ry7_int,
	__global double* ry1_old,
	__global double* ry2_old,
	__global double* ry3_old,
	__global double* ry4_old,
	__global double* ry5_old,
	__global double* ry6_old,
	__global double* ry7_old,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* ro_int,
	__global double* u_intt,
	__global double* rh_int,
	__global double* S,
	const int N) {
	int i = get_global_id(0);

	if ((i >= 0) && (i <= N)) {
		ro_int[i] = 0.;
		u_intt[i] = 0.;
		rh_int[i] = 0.;
		ry1_int[i] = 0.;
		ry2_int[i] = 0.;
		ry3_int[i] = 0.;
		ry4_int[i] = 0.;
		ry5_int[i] = 0.;
		ry6_int[i] = 0.;
		ry7_int[i] = 0.;
	}
	if ((i >= 1) && (i <= N)) {
		ry1_old[i] = ry1[i];
		ry2_old[i] = ry2[i];
		ry3_old[i] = ry3[i];
		ry4_old[i] = ry4[i];
		ry5_old[i] = ry5[i];
		ry6_old[i] = ry6[i];
		ry7_old[i] = ry7[i];
	}
}
__kernel void calc_int(__global double* u,
	__global double* ro,
	__global double* rh,
	__global double* pi,
	__global double* temp,
	__global double* u_intt,
	__global double* rh_int,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* ry1_int,
	__global double* ry2_int,
	__global double* ry3_int,
	__global double* ry4_int,
	__global double* ry5_int,
	__global double* ry6_int,
	__global double* ry7_int,
	__global double* k_TCP1,
	__global double* k_TCP2,
	__global double* k_TCP3,
	__global double* k_TCP4,
	__global double* k_TCP5,
	__global double* k_TCP6,
	__global double* k_TCP7,
	__global double* Mm,
	__global double* KP0,
	__global double* ML0,
	__global double* h0,
	const int N,
	const int OsnV,
	const int nMat,
	const int n_TCP,
	const double H,
	const double P0)
 {
	
	int i = get_global_id(0);
	if ((i >= 0) && (i <= N)) {
		double yl_local[7];
		double yr_local[7];
		double Dm[7];
		double Dml[7];
		double Dmr[7];
		double hir[7];
		double hil[7];
		yl_local[0] = ry1[i] / ro[i];
		yl_local[1] = ry2[i] / ro[i];
		yl_local[2] = ry3[i] / ro[i];
		yl_local[3] = ry4[i] / ro[i];
		yl_local[4] = ry5[i] / ro[i];
		yl_local[5] = ry6[i] / ro[i];
		yl_local[6] = ry7[i] / ro[i];

		yr_local[0] = ry1[i + 1] / ro[i + 1];
		yr_local[1] = ry2[i + 1] / ro[i + 1];
		yr_local[2] = ry3[i + 1] / ro[i + 1];
		yr_local[3] = ry4[i + 1] / ro[i + 1];
		yr_local[4] = ry5[i + 1] / ro[i + 1];
		yr_local[5] = ry6[i + 1] / ro[i + 1];
		yr_local[6] = ry7[i + 1] / ro[i + 1];
		
		double rl = ro[i];
		double ul = u[i];
		double hl = rh[i] / ro[i];
		double pil = pi[i];
		double templ = temp[i];
		double rr = ro[i + 1];
		double ur = u[i + 1];
		double hr = rh[i + 1] / ro[i + 1];
		double pir = pi[i + 1];
		double tempr = temp[i + 1];
		double alpha = ((fabs(ul)) > (fabs(ur)) ? (fabs(ul)) : (fabs(ur)));;
		double fr = (0.5 * (rl * ul + rr * ur) - 0.5 * alpha * (ro[i + 1] - ro[i]));
		double fry1 = 0.5 * (rl * ul * yl_local[0] + rr * ur * yr_local[0]) - 0.5 * alpha * (rr * yr_local[0] - rl * yl_local[0]);
		double fry2 = 0.5 * (rl * ul * yl_local[1] + rr * ur * yr_local[1]) - 0.5 * alpha * (rr * yr_local[1] - rl * yl_local[1]);
		double fry3 = 0.5 * (rl * ul * yl_local[2] + rr * ur * yr_local[2]) - 0.5 * alpha * (rr * yr_local[2] - rl * yl_local[2]);
		double fry4 = 0.5 * (rl * ul * yl_local[3] + rr * ur * yr_local[3]) - 0.5 * alpha * (rr * yr_local[3] - rl * yl_local[3]);
		double fry5 = 0.5 * (rl * ul * yl_local[4] + rr * ur * yr_local[4]) - 0.5 * alpha * (rr * yr_local[4] - rl * yl_local[4]);
		double fry6 = 0.5 * (rl * ul * yl_local[5] + rr * ur * yr_local[5]) - 0.5 * alpha * (rr * yr_local[5] - rl * yl_local[5]);
		double fry7 = 0.5 * (rl * ul * yl_local[6] + rr * ur * yr_local[6]) - 0.5 * alpha * (rr * yr_local[6] - rl * yl_local[6]);
		double fu = (0.5 * (ul * ul + ur * ur) - 0.5 * alpha * (u[i + 1] - u[i]));
		double fh = (0.5 * (rl * ul * hl + rr * ur * hr) - 0.5 * alpha * (rh[i + 1] - rh[i]));
		double QF = 0.5 * (Calc_KP_Nv(yr_local, Mm, tempr, nMat, OsnV, KP0) + Calc_KP_Nv(yl_local, Mm, templ, nMat, OsnV, KP0)) * (tempr - templ) / H;
		double MF = 0.5 * (Calc_ML_Nv(yr_local, Mm, tempr, nMat, OsnV, ML0) + Calc_ML_Nv(yl_local, Mm, templ, nMat, OsnV, ML0)) * (ur - ul) / H;

		calc_Dm(yl_local, (P0 + pil), templ,
			Dml, Mm, nMat);
		calc_Dm(yr_local, (P0 + pir), tempr,
			Dmr, Mm, nMat);

		for (int iM = 0; iM < nMat; iM++) {
			Dm[iM] = 0.5 * (rr * Dmr[iM] + rl * Dml[iM]) * (yr_local[iM] - yl_local[iM]) / H;
		}

		calc_h(h0, tempr, hir, k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);
		calc_h(h0, templ, hil, k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);

		double QDm = 0.;
		for (int iM = 0; iM < nMat; iM++) {
			QDm += 0.5 * (hir[iM] + hil[iM]) * Dm[iM];
		}

		ry1_int[i] += Dm[0] - fry1;
		ry2_int[i] += Dm[1] - fry2;
		ry3_int[i] += Dm[2] - fry3;
		ry4_int[i] += Dm[3] - fry4;
		ry5_int[i] += Dm[4] - fry5;
		ry6_int[i] += Dm[5] - fry6;
		ry7_int[i] += Dm[6] - fry7;
		u_intt[i] += (MF / rl) -fu;
		rh_int[i] += QDm + QF - fh;
	}
}
__kernel void calc_int1(__global double* u,
	__global double* ro,
	__global double* rh,
	__global double* pi,
	__global double* temp,
	__global double* u_intt,
	__global double* rh_int,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* ry1_int,
	__global double* ry2_int,
	__global double* ry3_int,
	__global double* ry4_int,
	__global double* ry5_int,
	__global double* ry6_int,
	__global double* ry7_int,
	__global double* k_TCP1,
	__global double* k_TCP2,
	__global double* k_TCP3,
	__global double* k_TCP4,
	__global double* k_TCP5,
	__global double* k_TCP6,
	__global double* k_TCP7,
	__global double* Mm,
	__global double* KP0,
	__global double* ML0,
	__global double* h0,
	const int N,
	const int OsnV,
	const int nMat,
	const int n_TCP,
	const double H,
	const double P0)
{

	int i = get_global_id(0);
	if ((i >= 0) && (i <= N)) {
		double yl_local[7];
		double yr_local[7];
		double Dm[7];
		double Dml[7];
		double Dmr[7];
		double hir[7];
		double hil[7];
		yl_local[0] = ry1[i] / ro[i];
		yl_local[1] = ry2[i] / ro[i];
		yl_local[2] = ry3[i] / ro[i];
		yl_local[3] = ry4[i] / ro[i];
		yl_local[4] = ry5[i] / ro[i];
		yl_local[5] = ry6[i] / ro[i];
		yl_local[6] = ry7[i] / ro[i];

		yr_local[0] = ry1[i + 1] / ro[i + 1];
		yr_local[1] = ry2[i + 1] / ro[i + 1];
		yr_local[2] = ry3[i + 1] / ro[i + 1];
		yr_local[3] = ry4[i + 1] / ro[i + 1];
		yr_local[4] = ry5[i + 1] / ro[i + 1];
		yr_local[5] = ry6[i + 1] / ro[i + 1];
		yr_local[6] = ry7[i + 1] / ro[i + 1];

		double rl = ro[i];
		double ul = u[i];
		double hl = rh[i] / ro[i];
		double pil = pi[i];
		double templ = temp[i];
		double rr = ro[i + 1];
		double ur = u[i + 1];
		double hr = rh[i + 1] / ro[i + 1];
		double pir = pi[i + 1];
		double tempr = temp[i + 1];
		double alpha = ((fabs(ul)) > (fabs(ur)) ? (fabs(ul)) : (fabs(ur)));;
		double fr = (0.5 * (rl * ul + rr * ur) - 0.5 * alpha * (ro[i + 1] - ro[i]));
		double fry1 = 0.5 * (rl * ul * yl_local[0] + rr * ur * yr_local[0]) - 0.5 * alpha * (rr * yr_local[0] - rl * yl_local[0]);
		double fry2 = 0.5 * (rl * ul * yl_local[1] + rr * ur * yr_local[1]) - 0.5 * alpha * (rr * yr_local[1] - rl * yl_local[1]);
		double fry3 = 0.5 * (rl * ul * yl_local[2] + rr * ur * yr_local[2]) - 0.5 * alpha * (rr * yr_local[2] - rl * yl_local[2]);
		double fry4 = 0.5 * (rl * ul * yl_local[3] + rr * ur * yr_local[3]) - 0.5 * alpha * (rr * yr_local[3] - rl * yl_local[3]);
		double fry5 = 0.5 * (rl * ul * yl_local[4] + rr * ur * yr_local[4]) - 0.5 * alpha * (rr * yr_local[4] - rl * yl_local[4]);
		double fry6 = 0.5 * (rl * ul * yl_local[5] + rr * ur * yr_local[5]) - 0.5 * alpha * (rr * yr_local[5] - rl * yl_local[5]);
		double fry7 = 0.5 * (rl * ul * yl_local[6] + rr * ur * yr_local[6]) - 0.5 * alpha * (rr * yr_local[6] - rl * yl_local[6]);
		double fu = (0.5 * (ul * ul + ur * ur) - 0.5 * alpha * (u[i + 1] - u[i]));
		double fh = (0.5 * (rl * ul * hl + rr * ur * hr) - 0.5 * alpha * (rh[i + 1] - rh[i]));
		double QF = 0.5 * (Calc_KP_Nv(yr_local, Mm, tempr, nMat, OsnV, KP0) + Calc_KP_Nv(yl_local, Mm, templ, nMat, OsnV, KP0)) * (tempr - templ) / H;
		double MF = 0.5 * (Calc_ML_Nv(yr_local, Mm, tempr, nMat, OsnV, ML0) + Calc_ML_Nv(yl_local, Mm, templ, nMat, OsnV, ML0)) * (ur - ul) / H;

		calc_Dm(yl_local, (P0 + pil), templ,
			Dml, Mm, nMat);
		calc_Dm(yr_local, (P0 + pir), tempr,
			Dmr, Mm, nMat);

		double roVc = 0.;
		for (int iM = 0; iM < nMat; iM++) {
			Dm[iM] = 0.5 * (rr * Dmr[iM] + rl * Dml[iM]) * (yr_local[iM] - yl_local[iM]) / H;
		}

		calc_h(h0, tempr, hir, k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);
		calc_h(h0, templ, hil, k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);
		
		double QDm = 0.;
		for (int iM = 0; iM < nMat; iM++) {
			QDm += 0.5 * (hir[iM] + hil[iM]) * Dm[iM];
		}
		ry1_int[i + 1] += fry1 - Dm[0];
		ry2_int[i + 1] += fry2 - Dm[1];
		ry3_int[i + 1] += fry3 - Dm[2];
		ry4_int[i + 1] += fry4 - Dm[3];
		ry5_int[i + 1] += fry5 - Dm[4];
		ry6_int[i + 1] += fry6 - Dm[5];
		ry7_int[i + 1] += fry7 - Dm[6];
		u_intt[i + 1] += fu - (MF / rr);
		rh_int[i + 1] += fh - QF - QDm;
	}
}
__kernel void calc_DTH(__global double* u,
	__global double* ro,
	__global double* rh,
	__global double* u_intt,
	__global double* rh_int,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* ry1_old,
	__global double* ry2_old,
	__global double* ry3_old,
	__global double* ry4_old,
	__global double* ry5_old,
	__global double* ry6_old,
	__global double* ry7_old,
	__global double* ry1_int,
	__global double* ry2_int,
	__global double* ry3_int,
	__global double* ry4_int,
	__global double* ry5_int,
	__global double* ry6_int,
	__global double* ry7_int,
	const int N,
	const double DTH) {
	int i = get_global_id(0);

	if ((i >= 1) && (i <= N)) {
		ry1[i] = ry1_old[i] + DTH * ry1_int[i];
		ry2[i] = ry2_old[i] + DTH * ry2_int[i];
		ry3[i] = ry3_old[i] + DTH * ry3_int[i];
		ry4[i] = ry4_old[i] + DTH * ry4_int[i];
		ry5[i] = ry5_old[i] + DTH * ry5_int[i];
		ry6[i] = ry6_old[i] + DTH * ry6_int[i];
		ry7[i] = ry7_old[i] + DTH * ry7_int[i];
		ro[i]  = ry1[i] + ry2[i] + ry3[i] + ry4[i] + ry5[i] + ry6[i] + ry7[i];
		u[i] += DTH * u_intt[i];
		rh[i] += DTH * rh_int[i];
	}
}
__kernel void calc_u(__global double* u,
	__global double* ro,
	__global double* pi,
	const int N,
	const double DT,
	const double H) {
	int i = get_global_id(0);

	if ((i >= 1) && (i <= N)) {
		u[i] -= DT * 0.5 * (pi[i + 1] - pi[i - 1]) / ro[i] / H;
	}
}
__kernel void calc_s(__global double* S,
	__global double* Mm,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* ro,
	__global double* pi,
	__global double* temp,
	__global double* k_TCP1,
	__global double* k_TCP2,
	__global double* k_TCP3,
	__global double* k_TCP4,
	__global double* k_TCP5,
	__global double* k_TCP6,
	__global double* k_TCP7,
	__global double* KP0,
	__global double* h0,
	__global double* Ri,
	const int N,
	const int OsnV,
	const int nMat,
	const int n_TCP,
	const double DT,
	const double H,
	const double P0) {
	int i = get_global_id(0);

	if ((i >= 1) && (i <= N)) {
		S[i] = 0.0;
		double yl[7];
		double yr[7];
		double y_m[7];

		double Dml[7];
		double Dmr[7];
		double Dm[7];

		double hil[7];
		double hir[7];
		double hii[7];

		yl[0] = ry1[i - 1] / ro[i - 1];
		yl[1] = ry2[i - 1] / ro[i - 1];
		yl[2] = ry3[i - 1] / ro[i - 1];
		yl[3] = ry4[i - 1] / ro[i - 1];
		yl[4] = ry5[i - 1] / ro[i - 1];
		yl[5] = ry6[i - 1] / ro[i - 1];
		yl[6] = ry7[i - 1] / ro[i - 1];

		yr[0] = ry1[i + 1] / ro[i + 1];
		yr[1] = ry2[i + 1] / ro[i + 1];
		yr[2] = ry3[i + 1] / ro[i + 1];
		yr[3] = ry4[i + 1] / ro[i + 1];
		yr[4] = ry5[i + 1] / ro[i + 1];
		yr[5] = ry6[i + 1] / ro[i + 1];
		yr[6] = ry7[i + 1] / ro[i + 1];

		y_m[0] = ry1[i] / ro[i];
		y_m[1] = ry2[i] / ro[i];
		y_m[2] = ry3[i] / ro[i];
		y_m[3] = ry4[i] / ro[i];
		y_m[4] = ry5[i] / ro[i];
		y_m[5] = ry6[i] / ro[i];
		y_m[6] = ry7[i] / ro[i];

		double fM = 0.0;
		for (int iM = 0; iM < nMat; iM++) fM += y_m[iM] / Mm[iM];
		fM = 1.0 / fM;
		
		double Cp = Calc_CP(y_m, temp[i], k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7);
		calc_h(h0, temp[i], hii, 
			k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);

		calc_h(h0, temp[i + 1], hir, 
			k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);

		calc_h(h0, temp[i - 1], hil, 
			k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);

		calc_Dm(y_m, P0 + pi[i], temp[i],
			Dm, Mm, nMat);

		calc_Dm(yr, P0 + pi[i + 1], temp[i + 1],
			Dmr, Mm, nMat);

		calc_Dm(yl, P0 + pi[i - 1], temp[i - 1],
			Dml, Mm, nMat);

		int index = nMat * (i - 1);
		double sumYh = 0.;
		double sumDF = 0.;
		double sumR = 0.;
		double sumQF = 0.;
		for (int im = 0; im < nMat; im++) sumR += (fM / Mm[im] - hii[im] / (Cp * temp[i])) * Ri[index + im];

		for (int im = 0; im < nMat; im++) sumYh += ro[i] * Dm[im] * (yr[im] - yl[im]) * (hir[im] - hil[im]) / (4. * H * H);
		
		for (int im = 0; im < nMat; im++) sumDF += (fM / Mm[im]) * (-0.5 * (ro[i] * Dm[im] + ro[i - 1] * Dml[im]) * (y_m[im] - yl[im]) / H + 0.5 * (ro[i + 1] * Dmr[im] + ro[i] * Dm[im]) * (yr[im] - y_m[im]) / H) / H;
		
		for(int im = 0; im < nMat; im++) sumQF += (-0.5 * (Calc_KP_Nv(yl, Mm, temp[i - 1], nMat, OsnV, KP0) + Calc_KP_Nv(y_m, Mm, temp[i], nMat, OsnV, KP0)) * 
			(temp[i] - temp[i - 1]) / H + 0.5 * 
			(Calc_KP_Nv(yr, Mm, temp[i + 1], nMat, OsnV, KP0) + Calc_KP_Nv(y_m, Mm, temp[i], nMat, OsnV, KP0)) * 
			(temp[i + 1] - temp[i]) / H) / H;

		S[i] += ((1. / (ro[i] * Cp * temp[i])) * sumQF) +
			((1. / (ro[i] * Cp * temp[i])) * sumYh + (1. / ro[i]) * sumDF) + 
			((1. / ro[i]) * sumR);
			
	}
}
__kernel void calc_urs(__global double* Mm,
	__global double* hii,
	__global double* urs,
	__global double* k_TCP1,
	__global double* k_TCP2,
	__global double* k_TCP3,
	__global double* k_TCP4,
	__global double* k_TCP5,
	__global double* k_TCP6,
	__global double* k_TCP7,
	__global double* h0,
	const double h,
	const double fT,
	const double P0,
	const double gR,
	const int nMat,
	const int n_TCP) {
	int iM = get_global_id(0);

	if (iM == 0) {
		double y_m[7];
		for (int iM = 0; iM < nMat; iM++) y_m[iM] = 0.;
		y_m[0] = 1.;
		urs[4] = 0.0;
		urs[0] = fT;
		urs[5] = fT;

		for (int i = 0; i < nMat; i++) {
			urs[4] += y_m[i] / Mm[i];
		}

		double fCP = 0.0;
		fCP += Calc_CP(y_m, urs[0], k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7);

		urs[7] = fCP;
		urs[6] = urs[7] - gR * urs[4];
		urs[1] = urs[7] / urs[6];
		double TRef = 298.15;
		double t1, t2;
		double hii_local[7];
		for (int i = 0; i < nMat; i++)
		{
			hii_local[i] = h0[i];
		}

		for (int j = 0; j < n_TCP; j++) {
			int q = j + 1;
			t2 = exp((q)*log(fT)) / (q);
			t1 = exp((q)*log(TRef)) / (q);

			hii_local[0] += k_TCP1[j] * (t2 - t1);
			hii_local[1] += k_TCP2[j] * (t2 - t1);
			hii_local[2] += k_TCP3[j] * (t2 - t1);
			hii_local[3] += k_TCP4[j] * (t2 - t1);
			hii_local[4] += k_TCP5[j] * (t2 - t1);
			hii_local[5] += k_TCP6[j] * (t2 - t1);
			hii_local[6] += k_TCP7[j] * (t2 - t1);

		}

		urs[3] = 0.0;

		for (int i = 0; i < nMat; i++) {

			urs[3] += y_m[i] * hii_local[i];
		}
		urs[2] = P0 / (gR * urs[0] * urs[4]);


	}
}
__kernel void bnd_cond(__global double* u,
	__global double* ro,
	__global double* rh,
	__global double* pi,
	__global double* temp,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* urs,
	const int N,
	const int nMat,
	const double temp_in,
	const double P0,
	const double u_g) {
	int iM = get_global_id(0);

	if (iM == 0) {
		double y_m[7];
		for (int iM = 0; iM < nMat; iM++) y_m[iM] = 0.;
		y_m[0] = 1.;
		temp[0] = temp_in;
		double p_g = P0 + pi[1];

		ro[0] = urs[2];
		u[0] = u_g;
		pi[0] = p_g - P0;
		rh[0] = urs[2] * urs[3];
		ry1[0] = urs[2] * y_m[0];
		ry2[0] = urs[2] * y_m[1];
		ry3[0] = urs[2] * y_m[2];
		ry4[0] = urs[2] * y_m[3];
		ry5[0] = urs[2] * y_m[4];
		ry6[0] = urs[2] * y_m[5];
		ry7[0] = urs[2] * y_m[6];

		ro[N + 1] = ro[N];
		u[N + 1] = u[N];
		rh[N + 1] = rh[N];
		temp[N + 1] = temp[N];
		pi[N + 1] = 0.;
		ry1[N + 1] = ry1[N];
		ry2[N + 1] = ry2[N];
		ry3[N + 1] = ry3[N];
		ry4[N + 1] = ry4[N];
		ry5[N + 1] = ry5[N];
		ry6[N + 1] = ry6[N];
		ry7[N + 1] = ry7[N];
		
	}
}
__kernel void calc_urs_mixt(__global double* k_TCP1,
	__global double* k_TCP2,
	__global double* k_TCP3,
	__global double* k_TCP4,
	__global double* k_TCP5,
	__global double* k_TCP6,
	__global double* k_TCP7,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* h0,
	__global double* rh,
	__global double* ro,
	__global double* temp,
	const double P0,
	const double gR,
	const int nMat,
	const int n_TCP,
	const int N) {
	int iM = get_global_id(0);

	if ((iM >= 1) && (iM <= N)) {
		double y_m[7];

		y_m[0] = ry1[iM] / ro[iM];
		y_m[1] = ry2[iM] / ro[iM];
		y_m[2] = ry3[iM] / ro[iM];
		y_m[3] = ry4[iM] / ro[iM];
		y_m[4] = ry5[iM] / ro[iM];
		y_m[5] = ry6[iM] / ro[iM];
		y_m[6] = ry7[iM] / ro[iM];

		double t1, t2;
		double fT = temp[iM];
		double hii[7];

		double fE = rh[iM] / ro[iM];
		for (int ic = 0; ic < 100; ic++)
		{
			double fCP = 0.0;

			calc_h(h0, fT, hii, k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7, nMat, n_TCP);

			for (int i = 0; i < nMat; i++) fCP += y_m[i] * hii[i];

			double fCP_ = Calc_CP(y_m, fT, k_TCP1, k_TCP2, k_TCP3, k_TCP4, k_TCP5, k_TCP6, k_TCP7);

			double fFT = fCP - fE;
			double fFT_ = fCP_;

			double fTg = fT - fFT / fFT_;
			if ((fTg - fT) * (fTg - fT) < 1.0e-10) { fT = fTg; break; }
			fT = fTg;
		}
		temp[iM] = fT;
	}
}
__kernel void calc_pi(__global double* u,
	__global double* ro,
	__global double* pi,
	__global double* S,
	const int N,
	const double DT,
	const double H) 
{

	int iM = get_global_id(0);
	if (iM == 0) {
		for (int iz = 0; iz < 10; iz++) {
			for (int i = 1; i <= N; i++) {
				pi[i] = 0.5 * (pi[i - 1] + pi[i + 1] - H * H * ro[i] * (0.5 * (u[i + 1] - u[i - 1]) / H - S[i]) / (DT));
			}
		}
	}

}
__kernel void calc_r(__global double* Ri,
	__global double* Mm,
	__global double* ry1,
	__global double* ry2,
	__global double* ry3,
	__global double* ry4,
	__global double* ry5,
	__global double* ry6,
	__global double* ry7,
	__global double* temp,
	__global double* A,
	__global double* En,
	const int N,
	const int nMat,
	const int nst,
	const double gR)
{
	int i = get_global_id(0);
	if ((i >= 1) && (i <= N)) {
		int index = nMat * (i - 1);

		double k[5];
		double xi[7];

		for (int ki = 0; ki < nst; ki++) k[ki] = A[ki] * exp(-En[ki] / (gR * temp[i]));

		xi[0] = ry1[i] / Mm[0];
		xi[1] = ry2[i] / Mm[1];
		xi[2] = ry3[i] / Mm[2];
		xi[3] = ry4[i] / Mm[3];
		xi[4] = ry5[i] / Mm[4];
		xi[5] = ry6[i] / Mm[5];
		xi[6] = ry7[i] / Mm[6];
		
		Ri[index] = Mm[0] * (-k[0] * xi[0] - k[1] * xi[0] * xi[4] - k[3] * xi[0] * xi[6] + k[4] * xi[5] * xi[5]);
		Ri[index + 1] = Mm[1] * (k[2] * xi[5] + k[4] * xi[5] * xi[5]); 
		Ri[index + 2] = Mm[2] * (k[3] * xi[6] * xi[0]);
		Ri[index + 3] = Mm[3] * (k[1] * xi[4] * xi[0]); 
		Ri[index + 4] = Mm[4] * (2 * k[0] * xi[0] - k[1] * xi[0] * xi[4]); 
		Ri[index + 5] = Mm[5] * (k[1] * xi[4] * xi[0] - k[2] * xi[5] + k[3] * xi[6] * xi[0] - 2 * k[4] * xi[5] * xi[5]); 
		Ri[index + 6] = Mm[6] * (k[2] * xi[5] - k[3] * xi[6] * xi[0]); 

	}
}



