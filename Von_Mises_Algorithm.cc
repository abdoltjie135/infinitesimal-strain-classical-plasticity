/* Author: Abdul Gafeeth Benjamin
 * Date: 23/08/2024
 * Location: Cape Town, South Africa
 * Description: This file contains the Von_mises algorithm which will later be placed in to 'plasticity_model.cc'.
 *              Algorithms were extracted from the textbook 'Computational Methods for Plasticity:
 *              Theory and Applications' by EA de Souza Neto et al. which is referred to as 'the textbook' in the
 *              comments.
 */


// include the necessary libraries
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <numeric>
#include <cmath>

#include <sys/stat.h>
#include <cerrno>

using namespace dealii;

// NOTE: Perfect plasticity can be solved in closed form
//  the hardening modulus is 0 for perfect plasticity
//  see section 7.3.4 of the textbook for Von Mises with perfect plasticity
// The following can be found in Box 7.3 and 7.4 from the textbook
bool return_mapping_and_derivative_stress_strain(const SymmetricTensor<2, dim>& elastic_strain_n,
 const SymmetricTensor<2, dim>& delta_strain, SymmetricTensor<4, dim>& stress_strain_tensor,
 std::shared_ptr<PointHistory<dim, double>> &qph, std::string yield_criteria, std::string hardening_law) const
{
 SymmetricTensor<2, dim> elastic_strain_trial = elastic_strain_n + delta_strain; // ε_n+1^trial

 auto elastic_strain_trial_eigenvector_pairs =
     compute_principal_values_vectors(elastic_strain_trial);
 auto elastic_strain_trial_eigenvalues = elastic_strain_trial_eigenvector_pairs.first;
 auto elastic_strain_trial_eigenvectors_matrix = elastic_strain_trial_eigenvector_pairs.second;

 double accumulated_plastic_strain_trial = accumulated_plastic_strain_n; // ε_p_n+1^trial
 SymmetricTensor<2, dim> deviatoric_stress_trial = 2 * shear_modulus *
     deviator(elastic_strain_trial);  // Deviatoric stress (Box 8.1, Step i) from the textbook
 double e_v_trial = trace(elastic_strain_trial); // Volumetric part of the trial elastic strain
 double p_trial = bulk_modulus * e_v_trial; // p_n+1^trial

 double q_trial = sqrt((3.0 / 2.0) * scalar_product(deviatoric_stress_trial, deviatoric_stress_trial)); // q_n+1^trial

 if (q_trial - yield_stress(yield_stress_0, hardening_modulus, accumulated_plastic_strain_trial) <= 0)
 {
     // TODO: Set values at n+1 to n+1^trial
 }
 else
 {
     tolerance = 1e-6; // Tolerance for the Newton-Raphson method
     // Newton-Raphson method
    for (unsigned int k = 0; k < 50; ++k)
    {
        double delta_sigma = 0.0;

        double phi = q_trial - yield_stress(yield_stress_0, hardening_modulus, accumulated_plastic_strain_n); // φ

        H = hardening_modulus; // Hardening modulus

        double d = -3 * shear_modulus - H; // d

        delta_sigma -= phi / d;

        phi = q_trial - 3 * shear_modulus * delta_sigma -
            yield_stress(yield_stress_0, hardening_modulus, accumulated_plastic_strain_n + delta_sigma); // φ

        if (abs(phi) < tolerance)
        {
            double p_n1 = p_trial;
            SymmetricTensor<2, dim> s_n1 = (1 - (delta_sigma * 3 * shear_modulus) / q_trial) * deviatoric_stress_trial;
            SymmetricTensor<2, dim> stress_n1 = s_n1 + p_n1 * identity_tensor<dim>();
            SymmetricTensor<2, dim> elastic_strain_n1 = (1 / (2 * shear_modulus)) * s_n1 +
                (1 / 3) * e_v_trial * IdentityTensor<2, dim>();
            double accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_sigma;
        }
    }
 }
}

// The following can be found in section 7.3.4 of the textbook
template <int dim>
void ConstitutiveLaw<dim>::consistent_tangent_operator(bool is_plastic, double delta_gamma,
    SymmetricTensor<2, dim> stress_n1) const
{
    // Initialize the tensor components
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                {
                    SymmetricTensor<4, dim> I_s[i][j][k][l] = 0.5 * ( (i == k && j == l) + (i == l && j == k) );
                }

    SymmetricTensor<4, dim> I_d = I_s - (1 / 3) * outer_product(identity_tensor<dim>(), identity_tensor<dim>());

    if (is_plastic == false)
    {
        SymmetricTensor<4, dim> D = 2 * shear_modulus * I_d +
            bulk_modulus * outer_product(identity_tensor<dim>(), identity_tensor<dim>());
    }
    else
    {
        SymmetricTensor<2, dim> N_n1 = deviatoric_stress_trial / deviatoric_stress_trial.norm();

        SymmetricTensor<4, dim> D = 2 * shear_modulus * (1 - (delta_sigma * 3 * shear_modulus) / q_trial) * I_d +
            6 * shear_modulus * pow(shear_modulus, 2) * (delta_sigma / q_trial - 1 / (3 * shear_modulus +
                hardening_modulus)) * outer_product(N_n1, N_n1) +
                    bulk_modulus * outer_product(identity_tensor<dim>(), identity_tensor<dim>());
    }
}

