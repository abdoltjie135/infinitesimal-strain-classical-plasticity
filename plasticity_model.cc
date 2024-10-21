/* Author: Abdul Gafeeth Benjamin
 * Date: 23/08/2024
 * Location: Cape Town, South Africa
 * Description: This file contains the implementation of a plasticity model
 *              and is for the partial completion of MEC 5068Z. Algorithms were extracted from the textbook
 *              'Computational Methods for Plasticity: Theory and Applications' by EA de Souza Neto et al.
 *              which is referred to as 'the textbook' in the comments.
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


namespace PlasticityModel
{
    using namespace dealii;

    // TODO: This needs to be setup to work with multiple cycles
    template <int dim>
    class PointHistory
    {
    public:
        PointHistory() = default;
        void setup(const SymmetricTensor<2, dim> &initial_stress);
        SymmetricTensor<2, dim> get_stress() const;
        void set_stress(const SymmetricTensor<2, dim> &stress);

        // TODO: I am not sure if these should be private or public
        SymmetricTensor<2, dim> elastic_strain;
        double accumulated_plastic_strain;
        double e_v1;
        double p1;
        SymmetricTensor<2, dim> deviatoric_stress;

    private:
        SymmetricTensor<2, dim> stress;
    };

    // TODO: I am not sure if the following function is needed
    template <int dim>
    void PointHistory<dim>::setup(const SymmetricTensor<2, dim> &initial_stress)
    {
        stress = initial_stress;
    }

    template <int dim>
    SymmetricTensor<2, dim> PointHistory<dim>::get_stress() const
    {
        return stress;
    }

    template <int dim>
    void PointHistory<dim>::set_stress(const SymmetricTensor<2, dim> &stress)
    {
        this->stress = stress;
    }

    template <int dim>
    class ConstitutiveLaw
    {
    public:
        ConstitutiveLaw(const double E,           // Young's modulus
                        const double nu,          // Poisson's ratio
                        const double sigma_0,     // initial yield stress
                        const double gamma_iso,   // isotropic hardening parameter
                        const double gamma_kin);  // kinematic hardening parameter

        void set_sigma_0(double sigma_zero);    // set the initial yield stress

        // The following function performs the return-mapping algorithm and determines the derivative of stress with
        // respect to strain based off of whether a one-vector or a two-vector return was used
        bool return_mapping_and_derivative_stress_strain(const SymmetricTensor<2, dim>& elastic_strain_n,
        const SymmetricTensor<2, dim>& delta_strain, SymmetricTensor<4, dim>& stress_strain_tensor,
        std::shared_ptr<PointHistory<dim>> &qph, std::string yield_criteria, std::string hardening_law) const;

        // FIXME: The following declaration along with the function definition below may not be correct
        void derivative_of_isotropic_tensor(Tensor<2, dim> X, Tensor<2, dim> Y, Tensor<2, dim> dy_dx,
            bool is_two_vector_return, bool is_right_corner) const;
        // X - stress tensor
        // Y - strain tensor
        // the type of return mapping is needed therefore the two boolean variables are being passed as variables

        void get_linearized_stress_strain_tensors(
            const SymmetricTensor<2, dim>& strain_tensor,
            SymmetricTensor<2, dim>& stress_tensor,
            SymmetricTensor<4, dim>& stress_strain_tensor_linearized,
            SymmetricTensor<4, dim>& stress_strain_tensor, std::string yield_criteria,
            std::string hardening_law) const;

    private:
        const double kappa;
        const double mu;
        const double E;
        const double nu;
        double sigma_0; // this has not been made constant because it will be adjusted later
        const double gamma_iso;
        const double gamma_kin;

        const SymmetricTensor<4, dim> stress_strain_tensor_kappa;
        const SymmetricTensor<4, dim> stress_strain_tensor_mu;
    };

    // The following is the definition of the constructor
    template <int dim>
    ConstitutiveLaw<dim>::ConstitutiveLaw(double E,
                                          double nu,
                                          double sigma_0,
                                          double gamma_iso,
                                          double gamma_kin)
    // initialize the member variables
        : kappa(E / (3 * (1 - 2 * nu)))
          , mu(E / (2 * (1 + nu)))
          , E(E)
          , nu(nu)
          , sigma_0(sigma_0)
          , gamma_iso(gamma_iso)
          , gamma_kin(gamma_kin)
          , stress_strain_tensor_kappa(kappa *
              outer_product(unit_symmetric_tensor<dim>(),
                            unit_symmetric_tensor<dim>()))
          , stress_strain_tensor_mu(
              2 * mu *
              (identity_tensor<dim>() - outer_product(unit_symmetric_tensor<dim>(),
                                                      unit_symmetric_tensor<dim>()) /
                  3.0))
    {
    } // constructor body is empty because all the work is done in the initializer list

    template <int dim>
    void ConstitutiveLaw<dim>::set_sigma_0(double sigma_zero)
    {
        sigma_0 = sigma_zero;
    }


    // Function to find and sort eigenvalues and eigenvectors of a tensor in descending order
    template <int dim>
    std::pair<std::array<double, dim>, Tensor<2, dim>> compute_principal_values_vectors(const SymmetricTensor<2, dim> &A)
    {
        // Find eigenvalues and eigenvectors
        auto eigenvector_pairs = eigenvectors(A);

        std::sort(eigenvector_pairs.begin(), eigenvector_pairs.end(),
                 [](const std::pair<double, SymmetricTensor<2, dim>> &a, const std::pair<double,
                     SymmetricTensor<2, dim>> &b) {
                     return a.first > b.first;
                 });

        Tensor<2, dim> sorted_eigenvectors_tensor;
        for (unsigned int i = 0; i < dim; ++i) {
            for (unsigned int j = 0; j < dim; ++j) {
                sorted_eigenvectors_tensor[j][i] = eigenvector_pairs[i].second[j];
            }
        }

        std::array<double, dim> sorted_eigenvalues(eigenvector_pairs.size());

        for (unsigned int i = 0; i < dim; ++i) {
            sorted_eigenvalues[i] = eigenvector_pairs[i].first;
        }

        return std::make_pair(sorted_eigenvalues, sorted_eigenvectors_tensor);
    }

    // Function to find the eigenvalues of a tensor and sort them in descending order
    template <int dim>
    std::array<double, dim> compute_principal_values(const SymmetricTensor<2, dim> &A)
    {
        // Find eigenvalues and eigenvectors
        auto eigenvalues_array = eigenvalues(A);

        std::sort (eigenvalues_array.begin(), eigenvalues_array.end(),
                   [](const double &a, const double &b) {
                       return a > b;
                   });

        return eigenvalues_array;
    }

    // TODO: Create a function that takes in an array of principal values and tensor of sorted eigenvectors and returns
    //  the tensor in the original co-ordinates system
    //  arrange the principal values on the diagonal of a tensor
    //  and then return a tensor that is the eigenvectors tensor times by principal values tensor times by the transpose
    //  of the eigenvectors tensor

    // Function to reconstruct the tensor from principal values and sorted eigenvectors in the original co-ordinate system
    template <int dim>
    SymmetricTensor<2, dim> reconstruct_tensor(const std::array<double, dim> &principal_values,
                                               const Tensor<2, dim> &eigenvectors_tensor)
    {
        SymmetricTensor<2, dim> reconstructed_tensor;

        // Compute the tensor T = V * D * V^T, where
        // V is the matrix of eigenvectors,
        // D is the diagonal matrix of principal values,
        // and V^T is the transpose of V.
        for (unsigned int i = 0; i < dim; ++i)
        {
            for (unsigned int j = i; j < dim; ++j) // Use symmetry: compute only for j >= i
            {
                double sum = 0.0;
                for (unsigned int k = 0; k < dim; ++k)
                {
                    sum += eigenvectors_tensor[i][k] * principal_values[k] * eigenvectors_tensor[j][k];
                }
                reconstructed_tensor[i][j] = sum;
            }
        }

        return reconstructed_tensor;
    }

    // Function which computes the yield stress
    double yield_stress(double sigma_y_old , double H, double epsilon_p)
    {
        return sigma_y_old + H * epsilon_p;
    }


    // TODO: Pass the quadrature point history as an argument to this function
    // TODO: The Von_Mises yield criteria should still be added
    //  both yield criteria should work with either isotropic or kinematic hardening

    template <int dim>
    bool ConstitutiveLaw<dim>::return_mapping_and_derivative_stress_strain(
        const SymmetricTensor<2, dim>& elastic_strain_n,
        const SymmetricTensor<2, dim>& delta_strain,
        SymmetricTensor<4, dim>& stress_strain_tensor,
        std::shared_ptr<PointHistory<dim>> &qph,
        std::string yield_criteria,
        std::string hardening_law) const
    {
        Assert(dim == 3, ExcNotImplemented());

        // Material properties for consistent tangent computation
        double young_modulus = E;
        double poisson_ratio = nu;
        const double yield_stress_0 = sigma_0;
        // TODO: Add some comments to relate code variable names to the ones in textbook
        double shear_modulus = mu;               // G in the textbook
        double bulk_modulus = kappa;             // K in the textbook
        const double r2g = 2.0 * shear_modulus;  // 2 * G
        const double r4g = 4.0 * shear_modulus;  // 4 * G
        const double hardening_modulus = gamma_iso; // H - this can not be const for non-linear isotropic hardening

        double accumulated_plastic_strain_n = 0.0;

        // The following boolean variables will be used later for construction of the consistent tangent matrix
        bool is_two_vector_return;
        bool is_right_corner;

        // Elastic Predictor Step (Box 8.1, Step i) from the textbook
        // TODO: Need to extract the eigenvectors of the following tensor because it is needed to contruct the stress
        //  tensor at the step n+1
        // TODO: The elastic trial strain will be passed as a parameter to this function
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

        // Spectral decomposition (Box 8.1, Step ii) from the textbook
        auto s_trial_eigenvalues = compute_principal_values(deviatoric_stress_trial);

        // TODO: Create a yield stress function

        // Plastic admissibility check (Box 8.1, Step iii) from the textbook
        double phi = s_trial_eigenvalues[0] - s_trial_eigenvalues[dim - 1] -
            yield_stress(yield_stress_0, hardening_modulus, accumulated_plastic_strain_trial);

        // declare deviatoric principal stresses
        double s1, s2, s3;

        // storing ds/de
        //  s is the deviatoric principal stresses
        //  e is the deviatoric elastic strain tensor
        // FIXME: Not sure if the following should be a symmetric tensor (this has been checked)
        Tensor<2, dim> ds_de;
        SymmetricTensor<2, dim> dsigma_de;

        if (phi <= 0)
        {
            // Elastic step
            // Set the values at n+1 to the trial values
            SymmetricTensor<2, dim> elastic_strain_n1 = elastic_strain_trial;
            double accumulated_plastic_strain_n1 = accumulated_plastic_strain_trial;
            double e_v_n1 = e_v_trial;
            double p_n1 = p_trial;
            SymmetricTensor<2, dim> deviatoric_stress_n1 = deviatoric_stress_trial;

            // TODO: save intermediate values to quadrature point history
            qph.elastic_strain = elastic_strain_n1;
            qph.accumulated_plastic_strain = accumulated_plastic_strain_n1;
            qph.e_v = e_v_n1;
            qph.p = p_n1;
            qph.deviatoric_stress = deviatoric_stress_n1;

            // FIXME: I think the following is exiting the function but not sure if this is being done correctly (this has been checked)
            return false;
        }

        // Plastic step: Return Mapping (Box 8.1, Step iv) from the textbook
        double delta_gamma = 0.0;
        double residual = s_trial_eigenvalues[0] - s_trial_eigenvalues[dim - 1] -
           yield_stress(yield_stress_0, hardening_modulus, accumulated_plastic_strain_n);
        double tolerance = 1e-10;
        bool valid_return = false;

        // Initially say it is going to be a two-vector return
        bool local_newton_converged_one_v = false;

        // TODO: You can solve this in closed form therefore we do not need the Newton loop
        //  can solve delta_gamma directly
        // One-vector return to main plane (Box 8.2) from the textbook
        for (unsigned int iteration = 0; iteration < 50; ++iteration)
        {
            // Update yield stress after accumulating plastic strain (Box 8.2, Step ii)
            // TODO: For general isotropic hardening the hardening slope (hardening_modulus) needs to be determined
            //  for linear isotropic hardening it is constant
            //  see box 8.2 in Computational Methods for Plasticity
            double residual_derivative = -4 * shear_modulus - hardening_modulus;
            double delta_gamma =- residual / residual_derivative;  // new guess for delta_gamma

            // Compute the residual (Box 8.2, Step ii)
            // FIXME: Not sure if my function for the residual is correct
            double residual = s_trial_eigenvalues[0] - s_trial_eigenvalues[dim - 1] - 4.0 * shear_modulus * delta_gamma
            - yield_stress(yield_stress_0, hardening_modulus,
                accumulated_plastic_strain_n + delta_gamma);;

            if (std::abs(residual) < tolerance)
            {
                // Update principal deviatoric stresses (Box 8.2, Step iii)
                s1 = s_trial_eigenvalues[0] - 2.0 * shear_modulus * delta_gamma;
                s2 = s_trial_eigenvalues[1];
                s3 = s_trial_eigenvalues[2] + 2.0 * shear_modulus * delta_gamma;

                local_newton_converged_one_v = true;

                break;
            }
        }

        // Checking if the Newton iteration converged
        AssertThrow(local_newton_converged_one_v,
            ExcMessage("Newton iteration did not converge for one-vector return"));

        // FIXME: I am not sure if the updated s values are accessible below
        // Check if the updated principal stresses satisfy s1 >= s2 >= s3 (Box 8.1, Step iv.b) from the textbook
        if (s1 >= s2 && s2 >= s3)
        {
            is_two_vector_return = false;

            double f = 2 * shear_modulus / (4 * shear_modulus + hardening_modulus);

            //  TODO: some of these are commented out becuase the tensor is initialized as 0
            // This relates to equation 8.4.1 in the textbook
            ds_de[0][0] = 2 * shear_modulus * (1 - f);
            ds_de[0][2] = 2 * shear_modulus * f;
            ds_de[1][1] = 2 * shear_modulus;
            ds_de[2][0] = 2 * shear_modulus * f;
            ds_de[2][2] = 2 * shear_modulus * (1 - f);

            // FIXME: I am not sure if the following will exit the function (this has been checked)
            // FIXME: Might not be good to exit the function at this point therefore no 'return' statement
            // return true;
        }
        else
        {
            is_two_vector_return = true;

            if (s_trial_eigenvalues[0] + s_trial_eigenvalues[dim - 1] - 2 * s_trial_eigenvalues[1] > 0)
            {
                // Right corner return
                is_right_corner = true;

                // computing ds/de for the two-vector right return
                // This relates to equation 8.5.2 in the textbook
                double daa = -4 * shear_modulus - hardening_modulus;
                double dab = -2 * shear_modulus - hardening_modulus;
                double dba = -2 * shear_modulus - hardening_modulus;
                double dbb = -4 * shear_modulus - hardening_modulus;

                double det_d = daa * dbb - dab * dba;

                ds_de[0][0] = 2 * shear_modulus * (1 - (8 * pow(shear_modulus, 2)) / det_d);
                ds_de[0][1] = (4 * pow(shear_modulus, 2) / det_d) *(dab - daa);
                ds_de[0][2] = (4 * pow(shear_modulus, 2) / det_d) * (dba - dbb);
                ds_de[1][0] = (8 * pow(shear_modulus, 3)) / det_d;
                ds_de[1][1] = 2 * shear_modulus * (1 + (2 * shear_modulus * daa) / det_d);
                ds_de[1][2] = -(4 * pow(shear_modulus, 2) / det_d) * dba;
                ds_de[2][0] = (8 * pow(shear_modulus, 3)) / det_d;
                ds_de[2][1] = -(4 * pow(shear_modulus, 2) / det_d) * dab;
                ds_de[2][2] = 2 * shear_modulus * (1 + (2 * shear_modulus * dbb) / det_d);
            }
            else
            {
                // Left corner return
                is_right_corner = false;

                // computing ds/de for the two-vector right return
                // This relates to equation 8.5.3 in the textbook
                double daa = -4 * shear_modulus - hardening_modulus;
                double dab = -2 * shear_modulus - hardening_modulus;
                double dba = -2 * shear_modulus - hardening_modulus;
                double dbb = -4 * shear_modulus - hardening_modulus;

                double det_d = daa * dbb - dab * dba;

                ds_de[0][0] = 2 * shear_modulus * (1 + (2 * shear_modulus * dbb) / det_d);
                ds_de[0][1] = -(4 * pow(shear_modulus, 2) / det_d) * dab;
                ds_de[0][2] = (8 * pow(shear_modulus, 3)) / det_d;
                ds_de[1][0] = -(4 * pow(shear_modulus, 2) / det_d) * dba;
                ds_de[1][1] = 2 * shear_modulus * (1 + (2 * shear_modulus * daa) / det_d);
                ds_de[1][2] = (8 * pow(shear_modulus, 3)) / det_d;
                ds_de[2][0] = (4 * pow(shear_modulus, 2) / det_d) * (dba - dbb);
                ds_de[2][1] = (4 * pow(shear_modulus, 2) / det_d) * (dab - daa);
                ds_de[2][2] = 2 * shear_modulus * (1 - (8 * pow(shear_modulus, 2)) / det_d);
            }

            Vector<double> delta_gamma_vector(2);
            delta_gamma_vector[0] = 0;
            delta_gamma_vector[1] = 0;

            double s_b = s_trial_eigenvalues[1] - s_trial_eigenvalues[2];

            if (is_right_corner)
            {
                s_b = s_trial_eigenvalues[0] - s_trial_eigenvalues[1];
            }

            double s_a = s_trial_eigenvalues[0] - s_trial_eigenvalues[2];

            // TODO: The function for the yield stress will look different for nonlinear isotropic hardening
            Vector<double> residual_vector(2);
            residual_vector[0] = s_a - yield_stress(yield_stress_0, hardening_modulus,
                accumulated_plastic_strain_n);
            residual_vector[1] = s_b - yield_stress(yield_stress_0, hardening_modulus,
                accumulated_plastic_strain_n);

            Vector<double> delta_gamma_vector_update(2);

            // Newton iteration for two-vector return (Box 8.3, Step ii) from the textbook
            for (unsigned int iteration = 0; iteration < 50; ++iteration)
            {
                double delta_gamma_sum = delta_gamma_vector[0] + delta_gamma_vector[1];
                double accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma_sum;

                // FIXME: The hardening slope is constant for linear isotropic hardening
                //  This will need to be made general later
                double H = hardening_modulus;

                // FIXME: I can not take the inverse of d_matrix
                FullMatrix<double> d_matrix(2, 2);
                d_matrix(0, 0) = -4.0 * shear_modulus - H;
                d_matrix(0, 1) = -2.0 * shear_modulus - H;
                d_matrix(1, 0) = -2.0 * shear_modulus - H;
                d_matrix(1, 1) = -4.0 * shear_modulus - H;

                FullMatrix<double> d_matrix_inverse(2, 2);
                d_matrix_inverse.invert(d_matrix);

                residual_vector *= -1;
                d_matrix_inverse.vmult(delta_gamma_vector_update, residual_vector);

                delta_gamma_vector += delta_gamma_vector_update;

                residual_vector[0] = s_a - 2 * shear_modulus * (2 * delta_gamma_vector[0] + delta_gamma_vector[1]) -
                    (yield_stress_0 + hardening_modulus * accumulated_plastic_strain_n1);
                residual_vector[1] = s_b - 2 * shear_modulus * (2 * delta_gamma_vector[1] + delta_gamma_vector[0]) -
                    (yield_stress_0 + hardening_modulus * accumulated_plastic_strain_n1);

                if (abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance)
                {
                    if (is_right_corner)
                    {
                        s1 = s_trial_eigenvalues[0] - 2 * shear_modulus * (delta_gamma_vector[0] +
                            delta_gamma_vector[1]);
                        s2 = s_trial_eigenvalues[1] + 2 * shear_modulus * delta_gamma_vector[1];
                        s3 = s_trial_eigenvalues[2] + 2 * shear_modulus * delta_gamma_vector[0];
                    }
                    else
                    {
                        s1 = s_trial_eigenvalues[0] - 2 * shear_modulus * delta_gamma_vector[0];
                        s2 = s_trial_eigenvalues[1] - 2 * shear_modulus * delta_gamma_vector[1];
                        s3 = s_trial_eigenvalues[2] + 2 * shear_modulus * (delta_gamma_vector[0] +
                            delta_gamma_vector[1]);
                    }

                    break;
                }
            }

            // TODO: Add the Assert to check if it converged
            //  not sure if it is correct
            AssertThrow(abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance,
                ExcMessage("Two-vector return did not converge"));
        }

        // TODO: All the common computations for the end should be moved here
        double p_n1 = p_trial;

        // deviatoric stress in the principal directions
        SymmetricTensor<2, dim> s_n1;
        s_n1[0][0] = s1;
        s_n1[1][1] = s2;
        s_n1[2][2] = s3;

        SymmetricTensor<2, dim> stress_n1 = s_n1 + p_n1 * identity_tensor<dim>();

        // TODO: The equation for the updated elastic strain needs to be completed
        //  the deviatoric principal stresses and directions are needed
        // TODO: Add the following to the quadrature point history
        //  when they are being outputted you want to output them in the reference configuration
        //  this would probably want to be done at the end of the function for all the variables

        SymmetricTensor<2, dim> elastic_strain_n1 = (1 / (2 * shear_modulus)) * s_n1 +
            (1.0 / 3.0) * e_v_trial * unit_symmetric_tensor<dim>();

        // Adding state variables to the quadrature point history
        // TODO: Change to forward arrows when writing to the shared pointer
        qph->elastic_strain = elastic_strain_n1;
        qph->stress = stress_n1;
        qph->p = p_n1;

        // FIXME: Ensure that these loops are working correctly
        // FIXME: move to the end
        // The brackets after the first for loop are not necessary
        // Equation 8.46 from textbook
        for (unsigned int i = 0; i < 3; ++i) // Iterate over i
        {
            for (unsigned int j = 0; j < 3; ++j) // Iterate over j
            {
                dsigma_de[i][j] = bulk_modulus;

                // Loop through the k index for the sum over k
                for (unsigned int k = 0; k < 3; ++k)
                {
                    double delta_kj = (k == j) ? 1.0 : 0.0;
                    dsigma_de[i][j] += ds_de[i][k] * (delta_kj - 1.0 / 3.0);
                }
            }
        }

        return true;
    }

    // NOTE: I am not sure if the following function should output the elasticity tensor if it is elastic
    // NOTE: I am wondering if elastic and elastoplastic consistent tangent operators should be passed as variables
    //  to the function
    // TODO: Send in the principal values and directions
    //  the bools will no longer be needed because they are dependant on the return-mapping chosen
    //  the function should output the 4th order tensor instead of the void
    template <int dim>
    void ConstitutiveLaw<dim>::derivative_of_isotropic_tensor(
        Tensor<2, dim> X, Tensor<2, dim> Y, Tensor<2, dim> dy_dx,
        bool is_two_vector_return, bool is_right_corner) const
    {
        // The following was extracted from appendix A in the textbook

        // FIXME: The eigenvalues and eigenvectors do not need to be computed again but can be passed as a variable
        //  to the function
        // The following functions which are used to get eigenvalues and eigenvectors
        // also sorts them in descending order
        auto x = compute_principal_values(X);                 // Eigenvalues from X
        auto y = compute_principal_values(Y);                 // Eigenvalues from Y
        auto e = compute_principal_values_vectors(X).second;  // Eigenvectors from X

        // Calculate the 4th-order tensor d[X^2]/dX_ijkl as per equation A.46 in the textbook
        Tensor<4, dim> dX2_dX;

        for (unsigned int i = 0; i < dim; ++i)
        {
            for (unsigned int j = 0; j < dim; ++j)
            {
                for (unsigned int k = 0; k < dim; ++k)
                {
                    for (unsigned int l = 0; l < dim; ++l)
                    {
                        // Calculate the Kronecker deltas inline
                        double delta_ik = (i == k) ? 1.0 : 0.0;
                        double delta_il = (i == l) ? 1.0 : 0.0;
                        double delta_jl = (j == l) ? 1.0 : 0.0;
                        double delta_kj = (k == j) ? 1.0 : 0.0;

                        // Apply the formula directly for the 4th-order tensor
                        dX2_dX[i][j][k][l] = 0.5 * (
                            delta_ik * X[l][j] +
                            delta_il * X[k][j] +
                            delta_jl * X[i][k] +
                            delta_kj * X[i][l]
                        );
                    }
                }
            }
        }

        // Compute the projection tensors Ei = ei ⊗ ei for each eigenvector
        Tensor<2, dim> E;
        for (unsigned int i = 0; i < dim; ++i)
        {
            E[i] = outer_product(e[i], e[i]);
        }

        Tensor<4, dim> D = 0;

        // The following can be found in box A.6 of the textbook
        if (x[0] != x[1] && x[1] != x[2])
        {
            for (unsigned int a = 0; a < dim; ++a)
            {
                unsigned int b;
                unsigned int c;

                if (a == 1)
                {
                    b = 2;
                    c = 3;
                }
                if (a == 2)
                {
                    b = 3;
                    c = 1;
                }
                if (a == 3)
                {
                    b = 1;
                    c = 2;
                }

                D += (y[a] / ((x[a] - x[b]) * (x[a] - x[c]))) * (dX2_dX - (x[b] - x[c]) * identity_tensor<dim>() -
                    ((x[a] - x[b]) + (x[a] - x[c])) * outer_product(E[a], E[a]) - (x[b] - x[c]) *
                    outer_product(E[b], E[b]) - outer_product(E[c], E[c]));
            }
            for (unsigned int i = 0; i < dim; ++i)
            {
                for (unsigned int j = 0; j < dim; ++j)
                {
                    D =+ dy_dx[i][j] * outer_product(E[i], E[i]);
                }
            }
        }
        else if (x[0] == x[1] || x[1] == x[2] || x[0] == x[2])
        {
            unsigned int a;
            unsigned int b;
            unsigned int c;

            if (x[0] == x[1])
            {
                a = 2;
                b = 0;
                c = 1;
            }
            if (x[1] == x[2])
            {
                a = 0;
                b = 1;
                c = 2;
            }

            // TODO: These need to be checked if they are mathematically correct
            double s1 = (y[0] - y[2]) / ((x[0] - x[2]) * (x[0] - x[2])) +
                        (1 / (x[0] - x[2])) * ((dy_dx[2][0] - dy_dx[2][1]) - (dy_dx[2][0] - dy_dx[2][2]));

            double s2 = 2 * x[2] * (y[0] - y[2]) / ((x[0] - x[2]) * (x[0] - x[2])) +
                       (x[0] + x[2]) / (x[0] - x[2]) * (dy_dx[2][0] - dy_dx[2][1]);

            double s3 = 2 * (y[0] - y[2]) / ((x[0] - x[2]) * (x[0] - x[2]) * (x[0] - x[2])) +
                       (1 / ((x[0] - x[2]) * (x[0] - x[2]))) *
                       ((dy_dx[2][0] + dy_dx[0][0]) + (dy_dx[0][0] - dy_dx[0][2]) - (dy_dx[2][0] - dy_dx[2][2]));

            double s4 = 2 * x[2] * (y[0] - y[2]) / ((x[0] - x[2]) * (x[0] - x[2]) * (x[0] - x[2])) +
                       (1 / ((x[0] - x[2]) * (x[0] - x[2]))) *
                       (dy_dx[2][0] - dy_dx[2][1]) + (x[2] / ((x[0] - x[2]) * (x[0] - x[2]))) *
                       ((dy_dx[2][0] + dy_dx[0][0]) + (dy_dx[0][0] - dy_dx[0][2]) - (dy_dx[2][0] - dy_dx[2][2]));

            double s5 = 2 * x[2] * (y[0] - y[2]) / ((x[0] - x[2]) * (x[0] - x[2]) * (x[0] - x[2])) +
                       (1 / ((x[0] - x[2]) * (x[0] - x[2]))) *
                       (dy_dx[2][0] - dy_dx[2][1]) + (x[2] / ((x[0] - x[2]) * (x[0] - x[2]))) *
                       ((dy_dx[2][0] + dy_dx[0][0]) + (dy_dx[0][0] - dy_dx[0][2]) - (dy_dx[2][0] - dy_dx[2][2]));

            double s6 = 2 * (x[2] * x[2]) * (y[0] - y[2]) / ((x[0] - x[2]) * (x[0] - x[2]) * (x[0] - x[2])) +
                       (x[0] * x[2]) / ((x[0] - x[2]) * (x[0] - x[2])) *
                       ((dy_dx[0][0] + dy_dx[2][0]) + (dy_dx[0][0] - dy_dx[0][2]) - (dy_dx[2][0] - dy_dx[2][2])) -
                       (x[2] * x[2]) / ((x[0] - x[2]) * (x[0] - x[2])) *
                       ((dy_dx[0][0] + dy_dx[2][0])) - (x[0] + x[2]) / (x[0] - x[2]) * dy_dx[2][0];

            D = s1 * dX2_dX - s2 * identity_tensor<dim>() - s3 * outer_product(x, x) +
                s4 * outer_product(x, unit_symmetric_tensor<dim>()) +
                    s5 * outer_product(unit_symmetric_tensor<dim>(), X) -
                    s6 * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
        }
        else if (x[0] == x[1] && x[1] == x[2])
        {
            D = (dy_dx[0][0] - dy_dx[0][1]) * identity_tensor<dim>() +
                dy_dx[0][1] * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
        }
    }


    template <int dim>
    void ConstitutiveLaw<dim>::get_linearized_stress_strain_tensors(
        const SymmetricTensor<2, dim>& strain_tensor,
        SymmetricTensor<2, dim>& stress_tensor,
        SymmetricTensor<4, dim>& stress_strain_tensor_linearized,
        SymmetricTensor<4, dim>& stress_strain_tensor,
        std::string yield_criteria,
        std::string hardening_law) const
    {
        Assert(dim == 3, ExcNotImplemented());

        stress_tensor = (stress_strain_tensor_kappa + stress_strain_tensor_mu) * strain_tensor;

        // Initialize tangent tensors as the elastic part
        stress_strain_tensor = stress_strain_tensor_mu;
        stress_strain_tensor_linearized = stress_strain_tensor_mu;

        // Material properties for consistent tangent computation
        double young_modulus = E;
        double poisson_ratio = nu;
        double yield_stress = sigma_0;
        double shear_modulus = mu;
        double bulk_modulus = kappa;
        double r2g = 2.0 * shear_modulus; // 2 * G
        double r4g = 4.0 * shear_modulus; // 4 * G

        // Create the second-order matrix for the tangent operator using Voigt notation
        FullMatrix<double> tangent_matrix(6, 6); // For Voigt notation representation in 3D

        if (yield_criteria == "Tresca")
        {
            SymmetricTensor<2, dim> s_tensor = deviator(stress_tensor); // Deviatoric stress tensor

            // Step 1: Compute the principal stresses and their corresponding directions
            auto s_tensor_principal = eigenvalues(s_tensor);
            std::vector<unsigned int> indices(s_tensor_principal.size());
            std::iota(indices.begin(), indices.end(), 0);

            // Sort indices based on eigenvalues to find min and max
            std::sort(indices.begin(), indices.end(), [&s_tensor_principal](unsigned int i1, unsigned int i2) {
                return s_tensor_principal[i1] < s_tensor_principal[i2];
            });

            double s1 = s_tensor_principal[indices[2]]; // Max principal stress
            double s2 = s_tensor_principal[indices[1]]; // Middle principal stress
            double s3 = s_tensor_principal[indices[0]]; // Min principal stress
            double max_shear_stress = 0.5 * (s1 - s3);

            // Check for plasticity based on the Tresca criterion
            if (max_shear_stress > (yield_stress / 2))
            {
                const double beta = (yield_stress / 2) / max_shear_stress;
                double delta_gamma = 0.0;
                double tolerance = 1e-10;
                bool valid_return = false;

                // Newton-Raphson iterative solver to find delta_gamma
                for (unsigned int iteration = 0; iteration < 50; ++iteration)
                {
                    double residual = (s1 - s3) - (yield_stress + delta_gamma * gamma_iso);

                    if (std::abs(residual) < tolerance)
                    {
                        valid_return = true;
                        break;
                    }

                    double dphi_dgamma = -gamma_iso - r4g;
                    delta_gamma -= residual / dphi_dgamma;
                }

                if (!valid_return)
                {
                    // Failed to converge, return
                    return;
                }

                // Update the principal deviatoric stresses using delta_gamma
                s1 -= r2g * delta_gamma;
                s3 += r2g * delta_gamma;

                // Determine if two-vector return is needed
                bool is_two_vector_return = !(s1 >= s2 && s2 >= s3);
                bool is_right_corner = (s1 + s3 > 2 * s2);

                // Step 2: Populate the tangent matrix based on return type for Tresca
                if (is_two_vector_return)
                {
                    double daa = r4g + gamma_iso;
                    double dab = r2g + gamma_iso;
                    double dba = r2g + gamma_iso;
                    double dbb = r4g + gamma_iso;
                    double det = daa * dbb - dab * dba;
                    double r2gdd = r2g / det;
                    double r4g2dd = r2g * r2gdd;

                    if (is_right_corner)
                    {
                        tangent_matrix(0, 0) = r2g * (1.0 - r2gdd * r4g);
                        tangent_matrix(0, 1) = r4g2dd * (daa - dab);
                        tangent_matrix(0, 2) = r4g2dd * (dbb - dba);
                        tangent_matrix(1, 0) = r4g2dd * dba;
                        tangent_matrix(1, 1) = r2g * (1.0 - r2gdd * daa);
                        tangent_matrix(1, 2) = r4g2dd * dab;
                        tangent_matrix(2, 0) = r4g2dd * dba;
                        tangent_matrix(2, 1) = r4g2dd * dab;
                        tangent_matrix(2, 2) = r2g * (1.0 - r2gdd * dbb);
                    }
                    else
                    {
                        tangent_matrix(0, 0) = r2g * (1.0 - r2gdd * dbb);
                        tangent_matrix(0, 1) = r4g2dd * dab;
                        tangent_matrix(0, 2) = r4g2dd * r2g;
                        tangent_matrix(1, 0) = r4g2dd * dba;
                        tangent_matrix(1, 1) = r2g * (1.0 - r2gdd * daa);
                        tangent_matrix(1, 2) = r4g2dd * r2g;
                        tangent_matrix(2, 0) = r4g2dd * (dbb - dba);
                        tangent_matrix(2, 1) = r4g2dd * (daa - dab);
                        tangent_matrix(2, 2) = r2g * (1.0 - r2gdd * r4g);
                    }
                }
                else
                {
                    // One-vector return (main plane return)
                    double factor = r2g / (r4g + gamma_iso);
                    tangent_matrix(0, 0) = r2g * (1.0 - factor);
                    tangent_matrix(0, 1) = 0.0;
                    tangent_matrix(0, 2) = r2g * factor;
                    tangent_matrix(1, 0) = 0.0;
                    tangent_matrix(1, 1) = r2g;
                    tangent_matrix(1, 2) = 0.0;
                    tangent_matrix(2, 0) = r2g * factor;
                    tangent_matrix(2, 1) = 0.0;
                    tangent_matrix(2, 2) = tangent_matrix(0, 0);
                }

                // Map the tangent matrix back to the fourth-order tensors
                map_voigt_to_tensor(tangent_matrix, stress_strain_tensor);
                map_voigt_to_tensor(tangent_matrix, stress_strain_tensor_linearized);
            }
            else
            {
                // Elastic step
                stress_strain_tensor = stress_strain_tensor_mu;
                stress_strain_tensor_linearized = stress_strain_tensor_mu;
            }

            // Add the bulk modulus components to both tangent tensors
            stress_strain_tensor += stress_strain_tensor_kappa;
            stress_strain_tensor_linearized += stress_strain_tensor_kappa;
        }
        else
        {
            AssertThrow(false, ExcNotImplemented());
        }
    }



    // template <int dim>
    // void ConstitutiveLaw<dim>::get_linearized_stress_strain_tensors(
    //     const SymmetricTensor<2, dim>& strain_tensor,
    //     SymmetricTensor<2, dim>& stress_tensor,
    //     SymmetricTensor<4, dim>& stress_strain_tensor_linearized,
    //     SymmetricTensor<4, dim>& stress_strain_tensor,
    //     std::string yield_criteria,
    //     std::string hardening_law) const
    // {
    //     Assert(dim == 3, ExcNotImplemented()); // checking if the code is being run in 3D
    //
    //     stress_tensor = (stress_strain_tensor_kappa + stress_strain_tensor_mu) * strain_tensor;
    //
    //     stress_strain_tensor = stress_strain_tensor_mu;
    //     stress_strain_tensor_linearized = stress_strain_tensor_mu;
    //
    //     if (yield_criteria == "Von-Mises")
    //     {
    //         SymmetricTensor<2, dim> deviator_stress_tensor = deviator(stress_tensor);
    //         const double deviator_stress_tensor_norm = deviator_stress_tensor.norm();
    //
    //         if (deviator_stress_tensor_norm > sigma_0)
    //         {
    //             const double beta = sigma_0 / deviator_stress_tensor_norm;
    //
    //             if (hardening_law == "isotropic")
    //             {
    //                 // Isotropic hardening
    //                 stress_strain_tensor *= (gamma_iso + (1 - gamma_iso) * beta);
    //                 stress_strain_tensor_linearized *= (gamma_iso + (1 - gamma_iso) * beta);
    //             }
    //             else if (hardening_law == "kinematic")
    //             {
    //                 // Kinematic hardening
    //                 backstress_tensor += gamma_kin * deviator(stress_tensor);
    //                 stress_tensor -= backstress_tensor; // Subtract backstress tensor
    //             }
    //
    //             deviator_stress_tensor /= deviator_stress_tensor_norm;
    //
    //             stress_strain_tensor_linearized -=
    //                 (1 - gamma_iso) * beta * 2 * mu * outer_product(deviator_stress_tensor, deviator_stress_tensor);
    //         }
    //
    //         stress_strain_tensor += stress_strain_tensor_kappa;
    //         stress_strain_tensor_linearized += stress_strain_tensor_kappa;
    //     }
    //     else if (yield_criteria == "Tresca")
    //     {
    //         auto principal_stresses = eigenvalues(stress_tensor);
    //         std::sort(principal_stresses.begin(), principal_stresses.end());
    //
    //         const double sigma_max = principal_stresses[dim - 1];
    //         const double sigma_min = principal_stresses[0];
    //         const double max_shear_stress = 0.5 * (sigma_max - sigma_min);
    //
    //         if (max_shear_stress > (sigma_0 / 2))
    //         {
    //             const double beta = (sigma_0 / 2) / max_shear_stress;
    //
    //             if (hardening_law == "isotropic")
    //             {
    //                 // Isotropic hardening
    //                 stress_strain_tensor *= (gamma_iso + (1 - gamma_iso) * beta);
    //                 stress_strain_tensor_linearized *= (gamma_iso + (1 - gamma_iso) * beta);
    //             }
    //             else if (hardening_law == "kinematic")
    //             {
    //                 // Kinematic hardening
    //                 backstress_tensor += gamma_kin * unit_symmetric_tensor<dim>() * (sigma_max - sigma_min);
    //                 stress_tensor -= backstress_tensor; // Subtract backstress tensor
    //             }
    //
    //             // Linearize based on the principal stresses
    //             SymmetricTensor<2, dim> principal_stress_tensor;
    //             for (unsigned int i = 0; i < principal_stresses.size(); ++i)
    //                 principal_stress_tensor[i][i] = principal_stresses[i];
    //
    //             stress_strain_tensor_linearized -=
    //                 (1 - gamma_iso) * beta * 2 * mu * outer_product(principal_stress_tensor, principal_stress_tensor);
    //         }
    //
    //         stress_strain_tensor += stress_strain_tensor_kappa;
    //         stress_strain_tensor_linearized += stress_strain_tensor_kappa;
    //     }
    //     else
    //     {
    //         AssertThrow(false, ExcNotImplemented());
    //     }
    // }


    namespace EquationData
    {
        template <int dim>
        class BoundaryForce : public Function<dim>
            // Function<dim> is a base class used to represent a scalar or vector-valued function that can be
            // defined over a domain
        {
        public:
            BoundaryForce();

            virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;

            virtual void vector_value(const Point<dim>& p, Vector<double>& values) const override;
        };

        template <int dim>
        BoundaryForce<dim>::BoundaryForce()  // constructor definition
            : Function<dim>(dim)             // initializing the base class constructor with the dimension
        {
        }

        // The following function returns the value of the boundary force (value) at a given point which is zero
        template <int dim>
        double BoundaryForce<dim>::value(const Point<dim>&, const unsigned int) const  // parameters are not
                                                                                       // named because they are
                                                                                       // not used
        {
            return 0.; // the boundary force is zero
        }

        // The following function determines the vector value of the boundary force at a given point
        template <int dim>
        void BoundaryForce<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const
        {
            for (unsigned int c = 0; c < this->n_components; ++c)
                values(c) = BoundaryForce<dim>::value(p, c);
        }

        template <int dim>
        class BoundaryValues : public Function<dim>
        {
        public:
            BoundaryValues();

            virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;
        };

        template <int dim>
        BoundaryValues<dim>::BoundaryValues()
            : Function<dim>(dim)
        {
        }

        template <int dim>
        double BoundaryValues<dim>::value(const Point<dim>& p, const unsigned int component) const
        {
            if (component == 2 && p[2] >= 1 - 1e-3) // if z and z = 1
                return 0.1;
            else
                return 0.;
        }
    }




    template <int dim>
    class PlasticityProblem
    {
    public:
        PlasticityProblem(const ParameterHandler& prm);  // constructor which initializes an instance of the class
                                                         // with the parameters in the prm object

        void run(); // function to run the simulation

        static void declare_parameters(ParameterHandler& prm);  // function to declare the parameters

    private:
        void make_grid();
        void setup_system();
        void compute_dirichlet_constraints();
        void assemble_newton_system(const TrilinosWrappers::MPI::Vector& linearization_point);
        void compute_nonlinear_residual(const TrilinosWrappers::MPI::Vector& linearization_point);
        void solve_newton_system();
        void solve_newton();
        void refine_grid();
        void move_mesh(const TrilinosWrappers::MPI::Vector& displacement) const;
        void output_results(const unsigned int current_refinement_cycle);

        MPI_Comm mpi_communicator;
        ConditionalOStream pcout;
        TimerOutput computing_timer;

        const unsigned int n_initial_global_refinements;
        parallel::distributed::Triangulation<dim> triangulation;  // distributing the mesh across multiple processors
                                                                  // for parallel computing

        std::vector<std::shared_ptr<PointHistory<dim>>> quadrature_point_history;

        const unsigned int fe_degree;
        const FESystem<dim> fe;
        const FE_Q<dim> fe_scalar;
        DoFHandler<dim> dof_handler;
        DoFHandler<dim> dof_handler_scalar;

        IndexSet locally_owned_dofs;
        IndexSet locally_owned_scalar_dofs;
        IndexSet locally_relevant_dofs;
        IndexSet locally_relevant_scalar_dofs;

        AffineConstraints<double> constraints_hanging_nodes;
        AffineConstraints<double> constraints_dirichlet_and_hanging_nodes;
        AffineConstraints<double> all_constraints;
        AffineConstraints<double> scalar_constraints;

        IndexSet active_set;  // not sure if this is needed for the plasticity code
        Vector<float> fraction_of_plastic_q_points_per_cell;

        std::vector<SymmetricTensor<2, dim>> stress_at_q_points;

        TrilinosWrappers::SparseMatrix newton_matrix;

        TrilinosWrappers::MPI::Vector solution;
        TrilinosWrappers::MPI::Vector newton_rhs;
        TrilinosWrappers::MPI::Vector newton_rhs_uncondensed;
        TrilinosWrappers::MPI::Vector diag_mass_matrix_vector;
        TrilinosWrappers::MPI::Vector stress_tensor_diagonal;
        TrilinosWrappers::MPI::Vector stress_tensor_off_diagonal;
        TrilinosWrappers::MPI::Vector stress_tensor_tmp;

        const double e_modulus, nu, gamma_iso, gamma_kin, sigma_0;
        ConstitutiveLaw<dim> constitutive_law;

        const std::string base_mesh;

        const double applied_displacement;

        const std::string yield_criteria;
        const std::string hardening_law;

        struct RefinementStrategy // a struct is similar to a class but with public members by default
        {
            enum value // enum is a user-defined data type that consists of int constants
            {
                refine_global,
                refine_percentage,
                refine_fix_dofs
            };
        };

        typename RefinementStrategy::value refinement_strategy;

        const bool transfer_solution;
        std::string output_dir;
        const unsigned int n_refinement_cycles;
        unsigned int current_refinement_cycle;
    };


    // I am not too sure if the indexing which follows is fine
    template <int dim>
    void PlasticityProblem<dim>::declare_parameters(ParameterHandler& prm)
    {
        prm.declare_entry(
            "polynomial degree",
            "1",
            Patterns::Integer(), // specifies that this parameter has to be an integer
            "Polynomial degree of the FE_Q finite element space, typically 1 or 2.");
        prm.declare_entry(
            "number of initial refinements",
            "2",
            Patterns::Integer(),
            "Number of initial global refinements steps before the first computation.");
        prm.declare_entry(
            "refinement strategy",
            "percentage",
            Patterns::Selection("global|percentage"), // this restricts the parameter to global or percentage
            "Mesh refinement strategy: \n"
            "global: one global refinement \n"
            "percentage: a fixed percentage of cells gets refined using Kelly error estimator.");
        prm.declare_entry(
            "number of cycles",
            "5",
            Patterns::Integer(),
            "Number of adaptive mesh refinement cycles.");
        prm.declare_entry(
            "output directory",
            "",
            Patterns::Anything(),
            "Directory for output files (graphical output and benchmark statistics. "
            "If empty, use the current directory.");
        prm.declare_entry(
            "transfer solution",
            "false",
            Patterns::Bool(),
            "Whether the solution should be used as a starting guess for the next finer mesh."
            "If false, then the iteration starts at zero on every mesh.");
        // The following will be adjusted to include the deformable bodies as required by the course
        prm.declare_entry(
            "base mesh",
            "box",
            Patterns::Selection("box|half sphere"),
            "Select the shape of the domain: 'box' or 'half sphere'.");
        // The following will control the displacement applied to the top face of the box
        prm.declare_entry(
            "applied displacement",
            "-0.002",
            Patterns::Double(),
            "Applied displacement to the top of the box");
        // The following parameter is for the yield-criteria
        prm.declare_entry(
            "yield-criteria",
            "Von-Mises",
            Patterns::Selection("Von-Mises|Tresca"),
            "Select the yield-criteria: 'Von-Mises' or 'Tresca'.");
        // The following parameter is for the hardening-law
        prm.declare_entry(
            "hardening-law",
            "isotropic",
            Patterns::Selection("isotropic|kinematic"),
            "Select the hardening-law: 'isotropic' or 'kinematic'.");
    }


    template <int dim>
    PlasticityProblem<dim>::PlasticityProblem(const ParameterHandler& prm)  // defining the constructor
        : mpi_communicator(MPI_COMM_WORLD)
          , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          // the following initializes a timer to track computation times
          , computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::never, TimerOutput::wall_times)

          , n_initial_global_refinements(prm.get_integer("number of initial refinements"))
          , triangulation(mpi_communicator)
          , fe_degree(prm.get_integer("polynomial degree"))
          , fe(FE_Q<dim>(QGaussLobatto<1>(fe_degree + 1)) ^ dim)
          , fe_scalar(fe_degree)
          // , fe_system(FE_Q<dim>(QGaussLobatto<1>(fe_degree + 1)) ^ (dim + 0.5 * dim * (dim + 1)))
          , dof_handler(triangulation)
          , dof_handler_scalar(triangulation)
          // , dof_handler_system(triangulation)

          , e_modulus(200000)
          , nu(0.3)
          , gamma_iso(0.01)
          , gamma_kin(0.01)
          , sigma_0(400.0)
          , constitutive_law(e_modulus, nu, sigma_0, gamma_iso, gamma_kin)

          , base_mesh(prm.get("base mesh"))

          , applied_displacement(prm.get_double("applied displacement"))

          , yield_criteria(prm.get("yield-criteria"))
          , hardening_law(prm.get("hardening-law"))

          , transfer_solution(prm.get_bool("transfer solution"))
          , n_refinement_cycles(prm.get_integer("number of cycles"))
          , current_refinement_cycle(0)

    {
        std::string strat = prm.get("refinement strategy");
        if (strat == "global")
            refinement_strategy = RefinementStrategy::refine_global;
        else if (strat == "percentage")
            refinement_strategy = RefinementStrategy::refine_percentage;
        else
            AssertThrow(false, ExcNotImplemented());

        output_dir = prm.get("output directory");
        if (output_dir != "" && *(output_dir.rbegin()) != '/')
            output_dir += "/";

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            const int ierr = mkdir(output_dir.c_str(), 0777);
            AssertThrow(ierr == 0 || errno == EEXIST, ExcIO());
        }

        pcout << "    Using output directory '" << output_dir << "'" << std::endl;
        pcout << "    FE degree " << fe_degree << std::endl;
        pcout << "    transfer solution " << (transfer_solution ? "true" : "false")
            << std::endl;


    }

    // The following is used to rotate the half-sphere
    Point<3> rotate_half_sphere(const Point<3>& in)
    {
        return {in[2], in[1], -in[0]};
    }


    // The following is used to generate the deformable meshes
    template <int dim>
    void PlasticityProblem<dim>::make_grid()
    {
        if (base_mesh == "half sphere")
        {
            const Point<dim> center(0, 0, 0);
            const double radius = 0.8;
            GridGenerator::half_hyper_ball(triangulation, center, radius);
            triangulation.reset_all_manifolds();

            GridTools::transform(&rotate_half_sphere, triangulation);
            GridTools::shift(Point<dim>(0.5, 0.5, 0.5), triangulation);

            SphericalManifold<dim> manifold_description(Point<dim>(0.5, 0.5, 0.5));
            GridTools::copy_boundary_to_manifold_id(triangulation);
            triangulation.set_manifold(0, manifold_description);
        }
        else
        {
            const Point<dim> p1(0, 0, 0);
            const Point<dim> p2(1.0, 1.0, 1.0);

            GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
        }

        triangulation.refine_global(n_initial_global_refinements);  // refine the mesh globally based on prm file
    }


    template <int dim>
    void PlasticityProblem<dim>::setup_system()
    {
        pcout << "Setting up system..." << std::endl;
        // setup dofs and get index sets for locally owned and relevant DoFs
        {
            TimerOutput::Scope t(computing_timer, "Setup: distribute DoFs");
            dof_handler.distribute_dofs(fe);
            dof_handler_scalar.distribute_dofs(fe_scalar);

            locally_owned_dofs = dof_handler.locally_owned_dofs();
            locally_owned_scalar_dofs = dof_handler_scalar.locally_owned_dofs();
            locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
            locally_relevant_scalar_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_scalar);
        }

        // setup hanging nodes and Dirichlet constraints
        {
            TimerOutput::Scope t(computing_timer, "Setup: constraints");
            constraints_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
            scalar_constraints.reinit(locally_owned_scalar_dofs, locally_relevant_scalar_dofs);
            DoFTools::make_hanging_node_constraints(dof_handler, constraints_hanging_nodes);


            pcout << "   Number of active cells: "
                << triangulation.n_global_active_cells() << std::endl
                << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

            compute_dirichlet_constraints();

            all_constraints.copy_from(constraints_dirichlet_and_hanging_nodes);
            all_constraints.close();
            scalar_constraints.close();

            constraints_hanging_nodes.close();
            constraints_dirichlet_and_hanging_nodes.close();
        }

        // Initialization of the vectors and the active set
        {
            TimerOutput::Scope t(computing_timer, "Setup: vectors");
            solution.reinit(locally_relevant_dofs, mpi_communicator);
            newton_rhs.reinit(locally_owned_dofs, mpi_communicator);
            newton_rhs_uncondensed.reinit(locally_owned_dofs, mpi_communicator);
            diag_mass_matrix_vector.reinit(locally_owned_dofs, mpi_communicator);
            fraction_of_plastic_q_points_per_cell.reinit(triangulation.n_active_cells());
            stress_tensor_diagonal.reinit(locally_owned_dofs, mpi_communicator);
            stress_tensor_off_diagonal.reinit(locally_owned_dofs, mpi_communicator);
            stress_tensor_tmp.reinit(locally_owned_scalar_dofs, mpi_communicator);
        }

        // Initialize the matrix structures that will be used in the simulation
        {
            TimerOutput::Scope t(computing_timer, "Setup: matrix");
            TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
                                                 mpi_communicator);

            DoFTools::make_sparsity_pattern(dof_handler,
                                            sp,
                                            constraints_dirichlet_and_hanging_nodes,
                                            false,
                                            Utilities::MPI::this_mpi_process(
                                                mpi_communicator));
            sp.compress();
            newton_matrix.reinit(sp);
        }
        // Initialize the quadrature formula
        const QGauss<dim> quadrature_formula(fe_degree + 1);

        pcout<<"Initializing quadrature point history..."<<std::endl;
        // Initializing quadrature point history
        quadrature_point_history.resize(triangulation.n_locally_owned_active_cells() * quadrature_formula.size());
        pcout<<"Quadrature point history resized."<<std::endl;

        // unsigned int history_index = 0;
        // for (const auto &cell : triangulation.active_cell_iterators())
        // {
        //     if (cell->is_locally_owned())
        //     {
        //         for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        //         {
        //             quadrature_point_history[history_index] = std::make_shared<PointHistory<dim, double>>();
        //             ++history_index;
        //         }
        //     }
        // }

        unsigned int local_history_index = 0;
        for (const auto &cell : triangulation.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                {
                    quadrature_point_history[local_history_index] = std::make_shared<PointHistory<dim>>();
                    ++local_history_index;
                }
            }
        }

        pcout<<"Quadrature point history initialized."<<std::endl;
    }


    // How this function works will probably have to be adjusted for the course
    template <int dim>
    void PlasticityProblem<dim>::compute_dirichlet_constraints()
    {
        constraints_dirichlet_and_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
        constraints_dirichlet_and_hanging_nodes.merge(constraints_hanging_nodes);

        if (base_mesh == "box")
        {
            const FEValuesExtractors::Scalar x_displacement(0);
            const FEValuesExtractors::Scalar y_displacement(1);
            const FEValuesExtractors::Scalar z_displacement(2);

            // the following function enforces dirichlet constrains over boundaries
            VectorTools::interpolate_boundary_values(
                dof_handler,
                // top face
                5,
                // EquationData::BoundaryValues<dim>(),
                Functions::ConstantFunction<dim>(applied_displacement, dim),
                constraints_dirichlet_and_hanging_nodes,
                fe.component_mask(z_displacement));

            VectorTools::interpolate_boundary_values(
                dof_handler,
                // bottom face
                4,
                Functions::ZeroFunction<dim>(dim),
                constraints_dirichlet_and_hanging_nodes,
                fe.component_mask(z_displacement));

            VectorTools::interpolate_boundary_values(
                dof_handler,
                // right face
                1,
                Functions::ZeroFunction<dim>(dim),
                constraints_dirichlet_and_hanging_nodes,
                fe.component_mask(y_displacement));

            VectorTools::interpolate_boundary_values(
                dof_handler,
                // left face
                0,
                Functions::ZeroFunction<dim>(dim),
                constraints_dirichlet_and_hanging_nodes,
                fe.component_mask(y_displacement));

            if (dim == 3)  // the front and back faces only exist in 3D
            {
                VectorTools::interpolate_boundary_values(
                    dof_handler,
                    // front face
                    3,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(x_displacement));

                VectorTools::interpolate_boundary_values(
                    dof_handler,
                    // back face
                    2,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(x_displacement));
            }
        }
        // the following is if the half-sphere is used
        else
            VectorTools::interpolate_boundary_values(
                dof_handler,
                0,
                EquationData::BoundaryValues<dim>(),
                constraints_dirichlet_and_hanging_nodes,
                ComponentMask());

        constraints_dirichlet_and_hanging_nodes.close();
    }


    template <int dim>
    void PlasticityProblem<dim>::assemble_newton_system(const TrilinosWrappers::MPI::Vector &linearization_point)
    {
        TimerOutput::Scope t(computing_timer, "Assembling");

        const QGauss<dim> quadrature_formula(fe_degree + 1);  // Use fe_degree for quadrature formula
        const QGauss<dim - 1> face_quadrature_formula(fe_degree + 1);

        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients | update_JxW_values);

        FEFaceValues<dim> fe_values_face(fe, face_quadrature_formula,
                                         update_values | update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);

        // Create a stress storage for each quadrature point
        std::vector<SymmetricTensor<2, dim>> stress_at_q_points(n_q_points);

        const FEValuesExtractors::Vector displacement(0);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                cell_matrix = 0;
                cell_rhs = 0;

                // Get strains at each quadrature point
                fe_values[displacement].get_function_symmetric_gradients(linearization_point, strain_tensor);

                // Retrieve the quadrature point history for this cell
                unsigned int cell_index = cell->active_cell_index() * n_q_points;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    SymmetricTensor<4, dim> stress_strain_tensor_linearized;
                    SymmetricTensor<4, dim> stress_strain_tensor;
                    SymmetricTensor<2, dim> stress_tensor;

                    // Compute the stress using the constitutive law
                    constitutive_law.get_linearized_stress_strain_tensors(
                        strain_tensor[q_point], stress_tensor, stress_strain_tensor_linearized,
                        stress_strain_tensor, yield_criteria, hardening_law);

                    // Store the computed stress in the PointHistory
                    quadrature_point_history[cell_index + q_point]->set_stress(stress_tensor);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const SymmetricTensor<2, dim> stress_phi_i =
                            stress_strain_tensor_linearized *
                                fe_values[displacement].symmetric_gradient(i, q_point);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            cell_matrix(i, j) += (stress_phi_i *
                                fe_values[displacement].symmetric_gradient(j, q_point) *
                                fe_values.JxW(q_point));
                        }

                        cell_rhs(i) += ((stress_phi_i - stress_strain_tensor *
                            fe_values[displacement].symmetric_gradient(i, q_point)) *
                            strain_tensor[q_point] * fe_values.JxW(q_point));
                    }
                }

                // Get the degrees of freedom indices and distribute local to global
                cell->get_dof_indices(local_dof_indices);
                all_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                    newton_matrix, newton_rhs, true);
            }
        }

        // Compress the matrix and right-hand side vector
        newton_matrix.compress(VectorOperation::add);
        newton_rhs.compress(VectorOperation::add);
    }


    // The following function calculates the residual vector for the nonlinear system of equations in the context
    // of a plasticity model. The residual represents the difference between the internal forces (arising from the
    // material's stress response) and the external forces (such as boundary conditions). The residual is crucial in
    // the Newton-Raphsonn method, which iteratively solves nonlinear systems by minimizing the residual.
    // The residual needs to be minimized due to balances of forces.
    template <int dim>
    void PlasticityProblem<dim>::compute_nonlinear_residual(const TrilinosWrappers::MPI::Vector& linearization_point)
    {
        const QGauss<dim> quadrature_formula(fe.degree + 1);
        const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_JxW_values);

        FEFaceValues<dim> fe_values_face(fe,
                                         face_quadrature_formula,
                                         update_values | update_quadrature_points |
                                         update_JxW_values);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const EquationData::BoundaryForce<dim> boundary_force;
        std::vector<Vector<double>> boundary_force_values(n_face_q_points, Vector<double>(dim));

        Vector<double> cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const FEValuesExtractors::Vector displacement(0);

        newton_rhs = 0;
        newton_rhs_uncondensed = 0;

        fraction_of_plastic_q_points_per_cell = 0;

        for (const auto& cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                cell_rhs = 0;

                std::vector<SymmetricTensor<2, dim>> strain_tensors(n_q_points);
                fe_values[displacement].get_function_symmetric_gradients(
                    linearization_point, strain_tensors);

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    SymmetricTensor<4, dim> stress_strain_tensor;
                    const bool q_point_is_plastic = constitutive_law.get_stress_strain_tensor(
                        strain_tensors[q_point], stress_strain_tensor, yield_criteria, hardening_law);
                    if (q_point_is_plastic)
                        ++fraction_of_plastic_q_points_per_cell(
                            cell->active_cell_index());

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        cell_rhs(i) -=
                        (strain_tensors[q_point] * stress_strain_tensor *
                            fe_values[displacement].symmetric_gradient(i, q_point) *
                            fe_values.JxW(q_point));

                        Tensor<1, dim> rhs_values;
                        rhs_values = 0; // this represents the boundary forces applied to the system
                        cell_rhs(i) += (fe_values[displacement].value(i, q_point) *
                            rhs_values * fe_values.JxW(q_point));
                    }
                }

                // distribute local to global
                cell->get_dof_indices(local_dof_indices);
                constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(
                    cell_rhs, local_dof_indices, newton_rhs);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    newton_rhs_uncondensed(local_dof_indices[i]) += cell_rhs(i);
            }

        fraction_of_plastic_q_points_per_cell /= quadrature_formula.size();
        newton_rhs.compress(VectorOperation::add);
        newton_rhs_uncondensed.compress(VectorOperation::add);
    }


    template <int dim>
    void PlasticityProblem<dim>::solve_newton_system()
    {
        TimerOutput::Scope t(computing_timer, "Solve");

        TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs,
                                                           mpi_communicator);
        // holds the solution that will be distributed across different processes
        // in parallel computing
        distributed_solution = solution;

        constraints_hanging_nodes.set_zero(distributed_solution);
        constraints_hanging_nodes.set_zero(newton_rhs);

        // The following is a preconditioner setup. This accelerates the convergence of the iterative solver.
        // It helps with improving the efficiency and stability of teh solution process, particularly for
        // large and complex systems.
        TrilinosWrappers::PreconditionAMG preconditioner; // AMG preconditioner
        {
            TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");

            std::vector<std::vector<bool>> constant_modes;
            DoFTools::extract_constant_modes(dof_handler,
                                             ComponentMask(),
                                             constant_modes);

            // configuring various parameters for the AMG preconditioner
            TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
            additional_data.constant_modes = constant_modes;
            additional_data.elliptic = true;
            additional_data.n_cycles = 1;
            additional_data.w_cycle = false;
            additional_data.output_details = false;
            additional_data.smoother_sweeps = 2;
            additional_data.aggregation_threshold = 1e-2;

            preconditioner.initialize(newton_matrix, additional_data);
        }

        // the following solves the linear system
        {
            TimerOutput::Scope t(computing_timer, "Solve: iterate");

            TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

            const double relative_accuracy = 1e-8;
            const double solver_tolerance = relative_accuracy *
                newton_matrix.residual(tmp, distributed_solution, newton_rhs);

            SolverControl solver_control(newton_matrix.m(), solver_tolerance);
            SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);
            // solve the system
            solver.solve(newton_matrix,
                         distributed_solution,
                         newton_rhs,
                         preconditioner);

            pcout << "         Error: " << solver_control.initial_value() << " -> "
                << solver_control.last_value() << " in "
                << solver_control.last_step() << " Bicgstab iterations."
                << std::endl;
        }

        all_constraints.distribute(distributed_solution);

        solution = distributed_solution;
    }


    template <int dim>
    void PlasticityProblem<dim>::solve_newton()
    {
        TrilinosWrappers::MPI::Vector old_solution(locally_owned_dofs, mpi_communicator);
        TrilinosWrappers::MPI::Vector residual(locally_owned_dofs, mpi_communicator);
        TrilinosWrappers::MPI::Vector tmp_vector(locally_owned_dofs, mpi_communicator);
        TrilinosWrappers::MPI::Vector locally_relevant_tmp_vector(locally_relevant_dofs, mpi_communicator);
        TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);

        double residual_norm;
        double previous_residual_norm = -std::numeric_limits<double>::max();

        const double correct_sigma = sigma_0;

        for (unsigned int newton_step = 1; newton_step <= 100; ++newton_step)
        {
            if (newton_step <= 2 &&
                ((transfer_solution && current_refinement_cycle == 0) ||
                    !transfer_solution))
                constitutive_law.set_sigma_0(correct_sigma);

            pcout << ' ' << std::endl;
            pcout << "   Newton iteration " << newton_step << std::endl;

            pcout << "      Assembling system... " << std::endl;
            newton_matrix = 0;
            newton_rhs = 0;
            assemble_newton_system(solution);

            pcout << "      Solving system... " << std::endl;
            solve_newton_system();

            if (newton_step == 1 ||
                (transfer_solution && newton_step == 2 && current_refinement_cycle == 0) ||
                (!transfer_solution && newton_step == 2))
            {
                compute_nonlinear_residual(solution);
                old_solution = solution;

                residual = newton_rhs;
                const unsigned int start_res = (residual.local_range().first),
                                   end_res = (residual.local_range().second);
                for (unsigned int n = start_res; n < end_res; ++n)
                    if (all_constraints.is_inhomogeneously_constrained(n))
                        residual(n) = 0;

                residual.compress(VectorOperation::insert);

                residual_norm = residual.l2_norm();

                pcout << "      Accepting Newton solution with residual: " << residual_norm << std::endl;
            }
            else
            {
                for (unsigned int i = 0; i < 5; ++i)
                {
                    distributed_solution = solution;

                    const double alpha = std::pow(0.5, static_cast<double>(i));
                    tmp_vector = old_solution;
                    tmp_vector.sadd(1 - alpha, alpha, distributed_solution);

                    TimerOutput::Scope t(computing_timer, "Residual and lambda");

                    locally_relevant_tmp_vector = tmp_vector;
                    compute_nonlinear_residual(locally_relevant_tmp_vector);
                    residual = newton_rhs;

                    const unsigned int start_res = (residual.local_range().first),
                                       end_res = (residual.local_range().second);
                    for (unsigned int n = start_res; n < end_res; ++n)
                        if (all_constraints.is_inhomogeneously_constrained(n))
                            residual(n) = 0;

                    residual.compress(VectorOperation::insert);

                    residual_norm = residual.l2_norm();

                    pcout
                        << "      Residual of the non-contact part of the system: "
                        << residual_norm << std::endl
                        << "         with a damping parameter alpha = " << alpha
                        << std::endl;

                    if (residual_norm < previous_residual_norm)
                        break;
                }

                solution = tmp_vector;
                old_solution = solution;
            }

            previous_residual_norm = residual_norm;

            if (residual_norm < 1e-10)
                break;
        }
    }


    // The following function is essential for adaptive meshing
    template <int dim>
    void PlasticityProblem<dim>::refine_grid()
    {
        if (refinement_strategy == RefinementStrategy::refine_global)
        {
            for (typename Triangulation<dim>::active_cell_iterator cell =
                     triangulation.begin_active();
                 cell != triangulation.end();
                 ++cell)
                if (cell->is_locally_owned())
                    cell->set_refine_flag();
        }
        else
        {
            Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
            KellyErrorEstimator<dim>::estimate(
                dof_handler,
                QGauss<dim - 1>(fe.degree + 2),
                std::map<types::boundary_id, const Function<dim>*>(),
                solution,
                estimated_error_per_cell);

            parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
                triangulation, estimated_error_per_cell, 0.3, 0.03);
        }

        triangulation.prepare_coarsening_and_refinement();

        parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
            solution_transfer(dof_handler);
        if (transfer_solution)
            solution_transfer.prepare_for_coarsening_and_refinement(solution);

        triangulation.execute_coarsening_and_refinement();

        setup_system();

        if (transfer_solution)
        {
            TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
            solution_transfer.interpolate(distributed_solution);

            constraints_hanging_nodes.distribute(distributed_solution);

            solution = distributed_solution;
            compute_nonlinear_residual(solution);
        }
    }


    // The following function updates the mesh geometry by displacing its vertices according ot the computed
    // displacement field
    template <int dim>
    void PlasticityProblem<dim>::move_mesh(const TrilinosWrappers::MPI::Vector& displacement) const
    {
        std::vector<bool> vertex_touched(triangulation.n_vertices(), false);

        // iterating over the cells
        for (const auto& cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
                // iterating over the vertices
                for (const auto v : cell->vertex_indices())
                    if (vertex_touched[cell->vertex_index(v)] == false)
                    {
                        vertex_touched[cell->vertex_index(v)] = true;

                        Point<dim> vertex_displacement;
                        for (unsigned int d = 0; d < dim; ++d)
                            vertex_displacement[d] = displacement(cell->vertex_dof_index(v, d));

                        cell->vertex(v) += vertex_displacement;
                    }
    }


    template <int dim>
    void PlasticityProblem<dim>::output_results(const unsigned int current_refinement_cycle)
    {
        TimerOutput::Scope t(computing_timer, "Graphical output");
        pcout << "      Writing graphical output... " << std::flush;

        pcout << "number of dofs scalar: " << dof_handler_scalar.n_dofs() << std::endl;
        pcout << "stress temp size: " << stress_tensor_tmp.size() << std::endl;

        // Move mesh
        move_mesh(solution);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        const QGauss<dim> quadrature_formula(fe.degree + 1);

        MappingQ1<dim> mapping;

        const std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(solution, std::vector<std::string>(dim, "displacement"),
                                 DataOut<dim>::type_dof_data, data_component_interpretation);
        data_out.add_data_vector(fraction_of_plastic_q_points_per_cell, "fraction_of_plastic_q_points");

        for (int i = 0; i < dim; ++i)
            for (int j = i; j < dim; ++j)
            {
                VectorTools::project(mapping, dof_handler_scalar, scalar_constraints, quadrature_formula,
                                     [&](const typename DoFHandler<dim>::active_cell_iterator &cell, const unsigned int q_point) -> double
                                     {
                                         unsigned int local_index = cell->active_cell_index() * quadrature_formula.size();
                                         const std::shared_ptr<PointHistory<dim>> &lqph = quadrature_point_history[local_index + q_point];
                                         const SymmetricTensor<2, dim> &T = lqph->get_stress();
                                         return T[i][j];
                                     },
                                     stress_tensor_tmp);

                std::string name = "T_" + std::to_string(i) + std::to_string(j);
                data_out.add_data_vector(dof_handler_scalar, stress_tensor_tmp, name);
            }

        data_out.build_patches();
        const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(output_dir,
            "solution", current_refinement_cycle, mpi_communicator, 2);
        pcout << pvtu_filename << std::endl;

        // Move mesh back
        TrilinosWrappers::MPI::Vector tmp(solution);
        tmp *= -1;
        move_mesh(tmp);
    }


    // The following function orchestrates the entire simulation. It manages mesh refinement, system solving
    // and output generation.
    template <int dim>
    void PlasticityProblem<dim>::run()
    {
        computing_timer.reset();
        for (; current_refinement_cycle < n_refinement_cycles; ++current_refinement_cycle)
        {
            {
                TimerOutput::Scope t(computing_timer, "Setup");

                pcout << std::endl;
                pcout << "Cycle " << current_refinement_cycle << ':' << std::endl;

                // initial grid setup
                if (current_refinement_cycle == 0)
                {
                    make_grid();
                    setup_system();
                }
                else
                // refine the grid if the current refinement cycle is not the first one
                {
                    TimerOutput::Scope t(computing_timer, "Setup: refine mesh");
                    refine_grid();
                }
            }

            solve_newton();

            output_results(current_refinement_cycle);

            computing_timer.print_summary();
            computing_timer.reset();

            Utilities::System::MemoryStats stats;
            Utilities::System::get_memory_stats(stats);
            pcout << "Peak virtual memory used, resident in kB: " << stats.VmSize
                << ' ' << stats.VmRSS << std::endl;
        }
    }
}


int main(int argc, char* argv[])
{
    using namespace dealii;
    using namespace PlasticityModel;

    try
    {
        ParameterHandler prm;
        PlasticityProblem<3>::declare_parameters(prm);
        if (argc != 2)
        {
            std::cerr << "*** Call this program as <./plasticity_model (parameter_file_name).prm>"
                << std::endl;
            return 1;
        }

        prm.parse_input(argv[1]);
        Utilities::MPI::MPI_InitFinalize mpi_initialization(
            argc, argv, numbers::invalid_unsigned_int);
        {
            PlasticityProblem<3> problem(prm);
            problem.run();
        }
    }
    catch (std::exception& exc)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }

    return 0;
}
