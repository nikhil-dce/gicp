/*************************************************************
  Generalized-ICP Copyright (c) 2009 Aleksandr Segal.
  All rights reserved.

  Redistribution and use in source and binary forms, with 
  or without modification, are permitted provided that the 
  following conditions are met:

 * Redistributions of source code must retain the above
  copyright notice, this list of conditions and the 
  following disclaimer.
 * Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the 
  following disclaimer in the documentation and/or other
  materials provided with the distribution.
 * The names of the contributors may not be used to endorse
  or promote products derived from this software
  without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGE.
 *************************************************************/



#include "gicp.h"
#include "optimize.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <fstream>

#include <iostream> //TODO: remove
#include <sstream>
#include <pthread.h>
#include <boost/format.hpp>

using namespace std;//TODO: remove

namespace dgc {
namespace gicp {

GICPPointSet::GICPPointSet()
{
	kdtree_points_ = NULL;
	kdtree_ = NULL;
	max_iteration_ = 200; // default value
	max_iteration_inner_ = 20; // default value for inner loop
	epsilon_ = 5e-4; // correspondes to ~1 mm (tolerence for convergence of GICP outer loop)
	epsilon_rot_ = 2e-3;
	//gicp_epsilon_ = .0004; // epsilon constant for gicp paper; this is NOT the convergence tolerence
	gicp_epsilon_ = .0004; // epsilon constant for gicp paper; this is NOT the convergence tolerence
	debug_ = false;
	solve_rotation_ = true;
	matrices_done_ = false;
	kdtree_done_ = false;
	flann_indexer_= NULL;
	flann_points_ = NULL;
	pthread_mutex_init(&mutex_, NULL);
}

GICPPointSet::~GICPPointSet()
{
	if (kdtree_ != NULL)
		delete kdtree_;
	if (kdtree_points_ != NULL)
		annDeallocPts(kdtree_points_);
}

void GICPPointSet::Clear(void) {
	pthread_mutex_lock(&mutex_);
	matrices_done_ = false;
	kdtree_done_ = false;
	if (kdtree_ != NULL) {
		delete kdtree_;
		kdtree_ = NULL;
	}
	if (kdtree_points_ != NULL) {
		annDeallocPts(kdtree_points_);
		kdtree_points_ = NULL;
	}
	point_.clear();
	pthread_mutex_unlock(&mutex_);

}

void GICPPointSet::BuildCudaKDTree(void)
{
	pthread_mutex_lock(&mutex_);

	if (kdtree_done_ ) {
		return;
	}

	kdtree_done_ = true;
	pthread_mutex_unlock(&mutex_);

	int i, n = NumPoints();

	if(n == 0) {
		return;
	}

	flann_points_  = new flann::Matrix<float> (new float[n*3], n, 3);

	for(size_t i = 0; i < n; ++i) {
		*(flann_points_->ptr()+i*3) = (float) point_[i].x;
		*(flann_points_->ptr() +i*3+1) = (float) point_[i].y;
		*(flann_points_->ptr() +i*3+2) = (float) point_[i].z;
	}

	//	flann_indices_= new flann::Matrix<int> (new int[n*3], n, 3);
	//	flann_distances_ = new flann::Matrix<float> (new float[n*3], n ,3);

//	 flann::KDTreeCuda3dIndex< flann::L2<float> > index(dataset);

	flann_indexer_ = new flann::KDTreeCuda3dIndex<flann::L2<float> > (*flann_points_);
//	flann_indexer_ = new flann::Index<flann::L2<float> > (*flann_points_, flann::KDTreeIndexParams(12));
	flann_indexer_->buildIndex();

	//	flann::Index<flann::L2<float> > indexA(datasetA, flann::KDTreeCuda3dIndexParams(64));
	//	indexA.buildIndex();

}

void GICPPointSet::BuildKDTree(void)
{
	pthread_mutex_lock(&mutex_);
	if(kdtree_done_) {
		return;
	}
	kdtree_done_ = true;
	pthread_mutex_unlock(&mutex_);

	int i, n = NumPoints();

	if(n == 0) {
		return;
	}

	kdtree_points_ = annAllocPts(n, 3);
	for(i = 0; i < n; i++) {
		kdtree_points_[i][0] = point_[i].x;
		kdtree_points_[i][1] = point_[i].y;
		kdtree_points_[i][2] = point_[i].z;
	}
	kdtree_ = new ANNkd_tree(kdtree_points_, n, 3, 10);
}

void GICPPointSet::ComputeCudaMatrices() {
	pthread_mutex_lock(&mutex_);
	if(flann_indexer_ == NULL) {
		return;
	}

	if(matrices_done_) {
		return;
	}

	matrices_done_ = true;
	pthread_mutex_unlock(&mutex_);

	int N  = NumPoints();
	int K = 20; // number of closest points to use for local covariance estimate
	double mean[3];

	// initialize with indices and distances flann:Matrix
	flann::Matrix<int> flann_search_indices(new int[(K*N)*3], (K*N), 3);
	flann::Matrix<float> flann_search_distances(new float[(K*N)*3], (K*N),3);

	gsl_vector *work = gsl_vector_alloc(3);
	if(work == NULL) {
		//TODO: handle
	}
	gsl_vector *gsl_singulars = gsl_vector_alloc(3);
	if(gsl_singulars == NULL) {
		//TODO: handle
	}
	gsl_matrix *gsl_v_mat = gsl_matrix_alloc(3, 3);
	if(gsl_v_mat == NULL) {
		//TODO: handle
	}

	flann_indexer_->knnSearch(*flann_points_, flann_search_indices, flann_search_distances, K, flann::SearchParams(16));

	// iterate through all points
	// iterate through all neighbors and do PCA

	for(int i = 0; i < N; i++) {

		gicp_mat_t &cov = point_[i].C;
		// zero out the cov and mean
		for(int k = 0; k < 3; k++) {
			mean[k] = 0.;
			for(int l = 0; l < 3; l++) {
				cov[k][l] = 0.;
			}
		}

		// find the covariance matrix
		for(int j = 0; j < K; j++) {
			GICPPoint &pt = point_[flann_search_indices[i][j]];

			mean[0] += pt.x;
			mean[1] += pt.y;
			mean[2] += pt.z;

			cov[0][0] += pt.x*pt.x;

			cov[1][0] += pt.y*pt.x;
			cov[1][1] += pt.y*pt.y;

			cov[2][0] += pt.z*pt.x;
			cov[2][1] += pt.z*pt.y;
			cov[2][2] += pt.z*pt.z;
		}

		mean[0] /= (double)K;
		mean[1] /= (double)K;
		mean[2] /= (double)K;
		// get the actual covariance
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l <= k; l++) {
				cov[k][l] /= (double)K;
				cov[k][l] -= mean[k]*mean[l];
				cov[l][k] = cov[k][l];
			}
		}

		// compute the SVD
		gsl_matrix_view gsl_cov = gsl_matrix_view_array(&cov[0][0], 3, 3);
		gsl_linalg_SV_decomp(&gsl_cov.matrix, gsl_v_mat, gsl_singulars, work);

		// zero out the cov matrix, since we know U = V since C is symmetric
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				cov[k][l] = 0;
			}
		}

		// reconstitute the covariance matrix with modified singular values using the column vectors in V.
		for(int k = 0; k < 3; k++) {
			gsl_vector_view col = gsl_matrix_column(gsl_v_mat, k);

			double v = 1.; // biggest 2 singular values replaced by 1
			if(k == 2) {   // smallest singular value replaced by gicp_epsilon
				v = gicp_epsilon_;
			}

			gsl_blas_dger(v, &col.vector, &col.vector, &gsl_cov.matrix);
		}
	}

	if(work != NULL) {
		gsl_vector_free(work);
	}
	if(gsl_v_mat != NULL) {
		gsl_matrix_free(gsl_v_mat);
	}
	if(gsl_singulars != NULL) {
		gsl_vector_free(gsl_singulars);
	}

	//if (flann_search_distances != NULL)
	delete[] flann_search_distances.ptr();

	//if (flann_search_indices != NULL)
	delete[] flann_search_indices.ptr();
}

void GICPPointSet::ComputeMatrices() {
	pthread_mutex_lock(&mutex_);
	if(kdtree_ == NULL) {
		return;
	}
	if(matrices_done_) {
		return;
	}
	matrices_done_ = true;
	pthread_mutex_unlock(&mutex_);

	int N  = NumPoints();
	int K = 20; // number of closest points to use for local covariance estimate
	double mean[3];

	ANNpoint query_point = annAllocPt(3);

	ANNdist *nn_dist_sq = new ANNdist[K];
	if(nn_dist_sq == NULL) {
		//TODO: handle this
	}
	ANNidx *nn_indecies = new ANNidx[K];
	if(nn_indecies == NULL) {
		//TODO: handle this
	}
	gsl_vector *work = gsl_vector_alloc(3);
	if(work == NULL) {
		//TODO: handle
	}
	gsl_vector *gsl_singulars = gsl_vector_alloc(3);
	if(gsl_singulars == NULL) {
		//TODO: handle
	}
	gsl_matrix *gsl_v_mat = gsl_matrix_alloc(3, 3);
	if(gsl_v_mat == NULL) {
		//TODO: handle
	}

	for(int i = 0; i < N; i++) {
		query_point[0] = point_[i].x;
		query_point[1] = point_[i].y;
		query_point[2] = point_[i].z;

		gicp_mat_t &cov = point_[i].C;
		// zero out the cov and mean
		for(int k = 0; k < 3; k++) {
			mean[k] = 0.;
			for(int l = 0; l < 3; l++) {
				cov[k][l] = 0.;
			}
		}

		kdtree_->annkSearch(query_point, K, nn_indecies, nn_dist_sq, 0.0);

		// find the covariance matrix
		for(int j = 0; j < K; j++) {
			GICPPoint &pt = point_[nn_indecies[j]];

			mean[0] += pt.x;
			mean[1] += pt.y;
			mean[2] += pt.z;

			cov[0][0] += pt.x*pt.x;

			cov[1][0] += pt.y*pt.x;
			cov[1][1] += pt.y*pt.y;

			cov[2][0] += pt.z*pt.x;
			cov[2][1] += pt.z*pt.y;
			cov[2][2] += pt.z*pt.z;
		}

		mean[0] /= (double)K;
		mean[1] /= (double)K;
		mean[2] /= (double)K;
		// get the actual covariance
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l <= k; l++) {
				cov[k][l] /= (double)K;
				cov[k][l] -= mean[k]*mean[l];
				cov[l][k] = cov[k][l];
			}
		}

		// compute the SVD
		gsl_matrix_view gsl_cov = gsl_matrix_view_array(&cov[0][0], 3, 3);
		gsl_linalg_SV_decomp(&gsl_cov.matrix, gsl_v_mat, gsl_singulars, work);

		// zero out the cov matrix, since we know U = V since C is symmetric
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				cov[k][l] = 0;
			}
		}

		// reconstitute the covariance matrix with modified singular values using the column vectors in V.
		for(int k = 0; k < 3; k++) {
			gsl_vector_view col = gsl_matrix_column(gsl_v_mat, k);

			double v = 1.; // biggest 2 singular values replaced by 1
			if(k == 2) {   // smallest singular value replaced by gicp_epsilon
				v = gicp_epsilon_;
			}

			gsl_blas_dger(v, &col.vector, &col.vector, &gsl_cov.matrix);
		}
	}

	if(nn_dist_sq != NULL) {
		delete [] nn_dist_sq;
	}
	if(nn_indecies != NULL) {
		delete [] nn_indecies;
	}
	if(work != NULL) {
		gsl_vector_free(work);
	}
	if(gsl_v_mat != NULL) {
		gsl_matrix_free(gsl_v_mat);
	}
	if(gsl_singulars != NULL) {
		gsl_vector_free(gsl_singulars);
	}
	query_point;
}

void GICPPointSet::costPlot(GICPPointSet *scan, dgc_transform_t base_t, dgc_transform_t t, double max_match_dist) {

	ofstream fout_cost("gicp_heading_cost_plot");

	double max_d_sq = pow(max_match_dist, 2);
	int num_matches = 0;
	int n = scan->NumPoints();
	double delta = 0.;
	dgc_transform_t t_last;
	ANNdist nn_dist_sq;
	ANNidx *nn_indecies = new ANNidx[n];
	ANNpoint query_point = annAllocPt(3);

	if(nn_indecies == NULL) {
		//TODO: fail here
	}

	gicp_mat_t *mahalanobis = new gicp_mat_t[n];
	if(mahalanobis == NULL) {
		//TODO: fail here
	}
	gsl_matrix *gsl_R = gsl_matrix_alloc(3, 3);
	if(gsl_R == NULL) {
		//TODO: fail here
	}
	gsl_matrix *gsl_temp = gsl_matrix_alloc(3, 3);
	if(gsl_temp == NULL) {
		//TODO: fail here
	}

	/* set up the optimization parameters */
	GICPOptData opt_data;
	opt_data.nn_indecies = nn_indecies;
	opt_data.p1 = scan;
	opt_data.p2 = this;
	opt_data.is_cuda = false;
	opt_data.M = mahalanobis;
	opt_data.solve_rotation = solve_rotation_;
	dgc_transform_copy(opt_data.base_t, base_t);

	GICPOptimizer opt;
	opt.SetDebug(debug_);

	/* set up the mahalanobis matricies */
	/* these are identity for now to ease debugging */
	for(int i = 0; i < n; i++) {
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				mahalanobis[i][k][l] = (k == l)?1:0.;
			}
		}
	}

	//	double tx(-0.476), ty(-1.168), tz(0.018), rx(0.0013), ry(0.0041), rz(0.0917);
	double tx, ty, tz, rx, ry, rz;
	dgc_transform_get_translation(t, &tx, &ty, &tz);
	dgc_transform_get_rotation_xyz(t, &rx, &ry, &rz);

	dgc_transform_print_console(t);
	dgc_transform_t base, dest;
	dgc_transform_identity(base);
	dgc_transform_identity(dest);

	//	dgc_transform_rotate_z(dest, rz);
	//	dgc_transform_rotate_y(dest, ry);
	//	dgc_transform_rotate_x(dest, rx);
	//	dgc_transform_print_console(dest);

	std::cout << "GT Heading: " << rz << std::endl;
	for (float heading = rz-0.6; heading <= rz+0.6; heading += 0.02) {

		//		dgc_transform_rpy(t,base, rx, ry, heading);
		dgc_transform_identity(t);
		dgc_transform_rotate_z(t, heading);
		dgc_transform_rotate_y(t, ry);
		dgc_transform_rotate_x(t, rx);
		t[0][3] = tx;
		t[1][3] = ty;
		t[2][3] = tz;

		if (fabs(heading - 0.479107) < 0.02)
			dgc_transform_print_console(t);

		//	return;

		//	double tx, ty;
		//	tx = t[0][3];
		//	ty = t[1][3];
		//	for (float x = tx-8; x <= tx+8; x+=0.2) {
		//
		//		for (float y = ty-8; y <= ty+8; y+=0.2) {

		//			// take the base transformation
		//			dgc_transform_identity(t);
		//			dgc_transform_copy(t, base_t);
		//			// apply the current state
		//
		//			gsl_vector_set(p, 0, x);
		//			gsl_vector_set(p, 1, y);
		//			gsl_vector_set(p, 2, z);
		//			gsl_vector_set(p, 3, roll);
		//			gsl_vector_set(p, 4, pitch);
		//			gsl_vector_set(p, 5, heading);
		//
		//			opt.apply_state_xyz_pub(t, p);
		//

		//			t[0][3] = x;
		//			t[1][3] = y;

		dgc_transform_t transform_R;
		dgc_transform_copy(transform_R, base_t);
		dgc_transform_left_multiply(transform_R, t);
		// copy the rotation component of the current total transformation (including base), into a gsl matrix
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				gsl_matrix_set(gsl_R, i, j, transform_R[i][j]);
			}
		}

		/* find correpondences */
		num_matches = 0;
		for (int i = 0; i < n; i++) {
			query_point[0] = scan->point_[i].x;
			query_point[1] = scan->point_[i].y;
			query_point[2] = scan->point_[i].z;

			dgc_transform_point(&query_point[0], &query_point[1],
					&query_point[2], base_t);
			dgc_transform_point(&query_point[0], &query_point[1],
					&query_point[2], t);

			kdtree_->annkSearch(query_point, 1, &nn_indecies[i], &nn_dist_sq, 0.0);

			if (nn_dist_sq < max_d_sq) {

				// set up the updated mahalanobis matrix here
				gsl_matrix_view C1 = gsl_matrix_view_array(&scan->point_[i].C[0][0], 3, 3);
				gsl_matrix_view C2 = gsl_matrix_view_array(&point_[nn_indecies[i]].C[0][0], 3, 3);
				gsl_matrix_view M = gsl_matrix_view_array(&mahalanobis[i][0][0], 3, 3);
				gsl_matrix_set_zero(&M.matrix);
				gsl_matrix_set_zero(gsl_temp);

				// M = R*C1  // using M as a temp variable here
				gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., gsl_R, &C1.matrix, 1., &M.matrix);

				// temp = M*R' // move the temp value to 'temp' here
				gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., &M.matrix, gsl_R, 0., gsl_temp);

				// temp += C2
				gsl_matrix_add(gsl_temp, &C2.matrix);
				// at this point temp = C2 + R*C1*R'

				// now invert temp to get the mahalanobis distance metric for gicp
				// M = temp^-1
				gsl_matrix_set_identity(&M.matrix);
				gsl_linalg_cholesky_decomp(gsl_temp);
				for(int k = 0; k < 3; k++) {
					gsl_vector_view row_view = gsl_matrix_row(&M.matrix, k);
					gsl_linalg_cholesky_svx(gsl_temp, &row_view.vector);
				}
				num_matches++;
			}
			else {
				nn_indecies[i] = -1; // no match
			}
		}

		opt_data.num_matches = num_matches;
		//			double cost = opt.Score(t, opt_data);
		double cost = opt.Score(t, opt_data);
		//			std::cout << "Cost: " << cost << std::endl;

		fout_cost << heading << ' ' << cost << '\n';
		//			//
		//		}
		//
		//		cout << "x: " << x << std::endl;
		cout << "Heading: " << heading << std::endl;
	}

	fout_cost.close();

}

int GICPPointSet::CudaAlignScan(GICPPointSet *scan, dgc_transform_t base_t, dgc_transform_t t, double max_match_dist, bool save_error_plot)
{
	double max_d_sq = pow(max_match_dist, 2);
	int num_matches = 0;
	int n = scan->NumPoints();
	double delta = 0.;
	dgc_transform_t t_last;
	ofstream fout_corresp;

	int K = 1; //NN
	// initialize with indices and distances flann:Matrix
	flann::Matrix<int> flann_search_indices(new int[(K*n)*3], (K*n), 3);
	flann::Matrix<float> flann_search_distances(new float[(K*n)*3], (K*n),3);

	gicp_mat_t *mahalanobis = new gicp_mat_t[n];
	if(mahalanobis == NULL) {
		//TODO: fail here
	}
	gsl_matrix *gsl_R = gsl_matrix_alloc(3, 3);
	if(gsl_R == NULL) {
		//TODO: fail here
	}
	gsl_matrix *gsl_temp = gsl_matrix_alloc(3, 3);
	if(gsl_temp == NULL) {
		//TODO: fail here
	}

	bool converged = false;
	int iteration = 0;
	bool opt_status = false;

	/* set up the optimization parameters */
	GICPOptData opt_data;
	opt_data.flann_search_nn_indices = &flann_search_indices;
	opt_data.nn_indecies = NULL;
	opt_data.p1 = scan;
	opt_data.p2 = this;
	opt_data.is_cuda = true;
	opt_data.M = mahalanobis;
	opt_data.solve_rotation = solve_rotation_;
	dgc_transform_copy(opt_data.base_t, base_t);

	GICPOptimizer opt;
	opt.SetDebug(debug_);
	opt.SetMaxIterations(max_iteration_inner_);
	/* set up the mahalanobis matricies */
	/* these are identity for now to ease debugging */
	for(int i = 0; i < n; i++) {
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				mahalanobis[i][k][l] = (k == l)?1:0.;
			}
		}
	}

	if(debug_) {
		dgc_transform_write(base_t, "t_base.tfm");
		dgc_transform_write(t, "t_0.tfm");
	}

	flann::Matrix<float> flann_query_points(new float[n*3], n, 3);

	while(!converged) {
		dgc_transform_t transform_R;
		dgc_transform_copy(transform_R, base_t);
		dgc_transform_left_multiply(transform_R, t);
		// copy the rotation component of the current total transformation (including base), into a gsl matrix
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				gsl_matrix_set(gsl_R, i, j, transform_R[i][j]);
			}
		}
		if(debug_) {
//			std::string correspondenceFile = (boost::format("correspondence.txt")%iteration).str();
			std::string correspondenceFile = "correspondence.txt";
			fout_corresp.open(correspondenceFile.c_str());
		}

		/* find correpondences */
		num_matches = 0;
		for (int i = 0; i < n; i++) {

			double query_x = scan->point_[i].x;
			double query_y = scan->point_[i].y;
			double query_z = scan->point_[i].z;

			dgc_transform_point(&query_x, &query_y,
					&query_z, base_t);
			dgc_transform_point(&query_x, &query_y,
					&query_z, t);

			*(flann_query_points.ptr()+i*3) = (float) query_x;
			*(flann_query_points.ptr()+i*3+1) = (float) query_y;
			*(flann_query_points.ptr()+i*3+2) = (float) query_z;
		}


		flann_indexer_->knnSearch(flann_query_points, flann_search_indices,
				flann_search_distances, 1, flann::SearchParams(16));

		//			kdtree_->annkSearch(query_point, 1, &nn_indecies[i], &nn_dist_sq, 0.0);
		for (int i = 0; i < n; i++) {

			double nn_dist_sq = flann_search_distances[i][0];

			if (nn_dist_sq < max_d_sq) {
				if(debug_) {
					fout_corresp << i << "\t" << flann_search_indices[i][0] << endl;
				}

				// set up the updated mahalanobis matrix here
				gsl_matrix_view C1 = gsl_matrix_view_array(&scan->point_[i].C[0][0], 3, 3);
				gsl_matrix_view C2 = gsl_matrix_view_array(&point_[flann_search_indices[i][0]].C[0][0], 3, 3);
				gsl_matrix_view M = gsl_matrix_view_array(&mahalanobis[i][0][0], 3, 3);
				gsl_matrix_set_zero(&M.matrix);
				gsl_matrix_set_zero(gsl_temp);

				// M = R*C1  // using M as a temp variable here
				gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., gsl_R, &C1.matrix, 1., &M.matrix);

				// temp = M*R' // move the temp value to 'temp' here
				gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., &M.matrix, gsl_R, 0., gsl_temp);

				// temp += C2
				gsl_matrix_add(gsl_temp, &C2.matrix);
				// at this point temp = C2 + R*C1*R'

				// now invert temp to get the mahalanobis distance metric for gicp
				// M = temp^-1
				gsl_matrix_set_identity(&M.matrix);
				gsl_linalg_cholesky_decomp(gsl_temp);
				for(int k = 0; k < 3; k++) {
					gsl_vector_view row_view = gsl_matrix_row(&M.matrix, k);
					gsl_linalg_cholesky_svx(gsl_temp, &row_view.vector);
				}
				num_matches++;
			}
			else {
				flann_search_indices[i][0] = -1; // no match
			}

		}

		if(debug_) { // save the current M matrices to file for debugging
			ofstream out("mahalanobis.txt");
			if(out) {
				for(int i = 0; i < n; i++) {
					for(int k = 0; k < 3; k++) {
						for(int l = 0; l < 3; l++) {
							out << mahalanobis[i][k][l] << "\t";
						}
					}
					out << endl;
				}
			}
			out.close();
		}

		if(debug_) {
			fout_corresp.close();
		}
		opt_data.num_matches = num_matches;

		/* optimize transformation using the current assignment and Mahalanobis metrics*/
		dgc_transform_copy(t_last, t);
		opt_status = opt.Optimize(t, opt_data);

		if(debug_) {
			cout << "Optimizer converged in " << opt.Iterations() << " iterations." << endl;
			cout << "Status: " << opt.Status() << endl;

			std::ostringstream filename;
			filename << "t_" << iteration+1 << ".tfm";
			dgc_transform_write(t, filename.str().c_str());
		}

		/* compute the delta from this iteration */
		delta = 0.;
		for(int k = 0; k < 4; k++) {
			for(int l = 0; l < 4; l++) {
				double ratio = 1;
				if(k < 3 && l < 3) { // rotation part of the transform
					ratio = 1./epsilon_rot_;
				}
				else {
					ratio = 1./epsilon_;
				}
				double c_delta = ratio*fabs(t_last[k][l] - t[k][l]);

				if(c_delta > delta) {
					delta = c_delta;
				}
			}
		}

		if(debug_) {
			cout << "delta = " << delta << endl;
		}

		/* check convergence */
		iteration++;
		if(iteration >= max_iteration_ || delta < 1) {
			converged = true;
		}
	}
	if(debug_) {
		cout << "Converged in " << iteration << " iterations." << endl;
		if(save_error_plot) {
			opt.PlotError(t, opt_data, "error_func");
		}
	}
	if(mahalanobis != NULL) {
		delete [] mahalanobis;
	}
	if(gsl_R != NULL) {
		gsl_matrix_free(gsl_R);
	}
	if(gsl_temp != NULL) {
		gsl_matrix_free(gsl_temp);
	}

	return iteration;
}


int GICPPointSet::AlignScan(GICPPointSet *scan, dgc_transform_t base_t, dgc_transform_t t, double max_match_dist, bool save_error_plot)
{
	double max_d_sq = pow(max_match_dist, 2);
	int num_matches = 0;
	int n = scan->NumPoints();
	double delta = 0.;
	dgc_transform_t t_last;
	ofstream fout_corresp;
	ANNdist nn_dist_sq;
	ANNidx *nn_indecies = new ANNidx[n];
	ANNpoint query_point = annAllocPt(3);

	if(nn_indecies == NULL) {
		//TODO: fail here
	}

	gicp_mat_t *mahalanobis = new gicp_mat_t[n];
	if(mahalanobis == NULL) {
		//TODO: fail here
	}
	gsl_matrix *gsl_R = gsl_matrix_alloc(3, 3);
	if(gsl_R == NULL) {
		//TODO: fail here
	}
	gsl_matrix *gsl_temp = gsl_matrix_alloc(3, 3);
	if(gsl_temp == NULL) {
		//TODO: fail here
	}

	bool converged = false;
	int iteration = 0;
	bool opt_status = false;


	/* set up the optimization parameters */
	GICPOptData opt_data;
	opt_data.flann_search_nn_indices = NULL;
	opt_data.nn_indecies = nn_indecies;
	opt_data.p1 = scan;
	opt_data.p2 = this;
	opt_data.is_cuda = false;
	opt_data.M = mahalanobis;
	opt_data.solve_rotation = solve_rotation_;
	dgc_transform_copy(opt_data.base_t, base_t);

	GICPOptimizer opt;
	opt.SetDebug(debug_);
	opt.SetMaxIterations(max_iteration_inner_);
	/* set up the mahalanobis matricies */
	/* these are identity for now to ease debugging */
	for(int i = 0; i < n; i++) {
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				mahalanobis[i][k][l] = (k == l)?1:0.;
			}
		}
	}

	if(debug_) {
		dgc_transform_write(base_t, "t_base.tfm");
		dgc_transform_write(t, "t_0.tfm");
	}

	while(!converged) {
		dgc_transform_t transform_R;
		dgc_transform_copy(transform_R, base_t);
		dgc_transform_left_multiply(transform_R, t);
		// copy the rotation component of the current total transformation (including base), into a gsl matrix
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				gsl_matrix_set(gsl_R, i, j, transform_R[i][j]);
			}
		}
		if(debug_) {
			fout_corresp.open("correspondence.txt");
		}
		/* find correpondences */
		num_matches = 0;
		for (int i = 0; i < n; i++) {
			query_point[0] = scan->point_[i].x;
			query_point[1] = scan->point_[i].y;
			query_point[2] = scan->point_[i].z;

			dgc_transform_point(&query_point[0], &query_point[1],
					&query_point[2], base_t);
			dgc_transform_point(&query_point[0], &query_point[1],
					&query_point[2], t);

			kdtree_->annkSearch(query_point, 1, &nn_indecies[i], &nn_dist_sq, 0.0);

			if (nn_dist_sq < max_d_sq) {
				if(debug_) {
					fout_corresp << i << "\t" << nn_indecies[i] << endl;
				}

				// set up the updated mahalanobis matrix here
				gsl_matrix_view C1 = gsl_matrix_view_array(&scan->point_[i].C[0][0], 3, 3);
				gsl_matrix_view C2 = gsl_matrix_view_array(&point_[nn_indecies[i]].C[0][0], 3, 3);
				gsl_matrix_view M = gsl_matrix_view_array(&mahalanobis[i][0][0], 3, 3);
				gsl_matrix_set_zero(&M.matrix);
				gsl_matrix_set_zero(gsl_temp);

				// M = R*C1  // using M as a temp variable here
				gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., gsl_R, &C1.matrix, 1., &M.matrix);

				// temp = M*R' // move the temp value to 'temp' here
				gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., &M.matrix, gsl_R, 0., gsl_temp);

				// temp += C2
				gsl_matrix_add(gsl_temp, &C2.matrix);
				// at this point temp = C2 + R*C1*R'

				// now invert temp to get the mahalanobis distance metric for gicp
				// M = temp^-1
				gsl_matrix_set_identity(&M.matrix);
				gsl_linalg_cholesky_decomp(gsl_temp);
				for(int k = 0; k < 3; k++) {
					gsl_vector_view row_view = gsl_matrix_row(&M.matrix, k);
					gsl_linalg_cholesky_svx(gsl_temp, &row_view.vector);
				}
				num_matches++;
			}
			else {
				nn_indecies[i] = -1; // no match
			}
		}

		if(debug_) { // save the current M matrices to file for debugging
			ofstream out("mahalanobis.txt");
			if(out) {
				for(int i = 0; i < n; i++) {
					for(int k = 0; k < 3; k++) {
						for(int l = 0; l < 3; l++) {
							out << mahalanobis[i][k][l] << "\t";
						}
					}
					out << endl;
				}
			}
			out.close();
		}

		if(debug_) {
			fout_corresp.close();
		}
		opt_data.num_matches = num_matches;

		/* optimize transformation using the current assignment and Mahalanobis metrics*/
		dgc_transform_copy(t_last, t);
		opt_status = opt.Optimize(t, opt_data);

		if(debug_) {
			cout << "Optimizer converged in " << opt.Iterations() << " iterations." << endl;
			cout << "Status: " << opt.Status() << endl;

			std::ostringstream filename;
			filename << "t_" << iteration+1 << ".tfm";
			dgc_transform_write(t, filename.str().c_str());
		}

		/* compute the delta from this iteration */
		delta = 0.;
		for(int k = 0; k < 4; k++) {
			for(int l = 0; l < 4; l++) {
				double ratio = 1;
				if(k < 3 && l < 3) { // rotation part of the transform
					ratio = 1./epsilon_rot_;
				}
				else {
					ratio = 1./epsilon_;
				}
				double c_delta = ratio*fabs(t_last[k][l] - t[k][l]);

				if(c_delta > delta) {
					delta = c_delta;
				}
			}
		}
		if(debug_) {
			cout << "delta = " << delta << endl;
		}

		/* check convergence */
		iteration++;
		if(iteration >= max_iteration_ || delta < 1) {
			converged = true;
		}
	}
	if(debug_) {
		cout << "Converged in " << iteration << " iterations." << endl;
		if(save_error_plot) {
			opt.PlotError(t, opt_data, "error_func");
		}
	}
	if(nn_indecies != NULL) {
		delete [] nn_indecies;
	}
	if(mahalanobis != NULL) {
		delete [] mahalanobis;
	}
	if(gsl_R != NULL) {
		gsl_matrix_free(gsl_R);
	}
	if(gsl_temp != NULL) {
		gsl_matrix_free(gsl_temp);
	}
	annDeallocPt(query_point);

	return iteration;
}
void GICPPointSet::SavePoints(const char *filename) {
	ofstream out(filename);

	if(out) {
		int n = NumPoints();
		for(int i = 0; i < n; i++) {
			out << point_[i].x << "\t" << point_[i].y << "\t" << point_[i].z << endl;
		}
	}
	out.close();
}

void GICPPointSet::SaveMatrices(const char *filename) {
	ofstream out(filename);

	if(out) {
		int n = NumPoints();
		for(int i = 0; i < n; i++) {
			for(int k = 0; k < 3; k++) {
				for(int l = 0; l < 3; l++) {
					out << point_[i].C[k][l] << "\t";
				}
			}
			out << endl;
		}
	}
	out.close();
}
}
}
