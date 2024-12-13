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
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
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


// TODO: remove the stuff related to the refinement strategy

namespace PlasticityModel
{
    using namespace dealii;

    template <int dim>
    class PointHistory
    {
    public:
        PointHistory(double K, double G);

        void store_internal_variables();

        // NOTE: back_stress should be part of this for kinematic hardening
        void set_stress(const SymmetricTensor<2, dim> &stress);
        void set_elastic_strain(const SymmetricTensor<2, dim> &elastic_strain);
        void set_strain(const SymmetricTensor<2, dim> &plastic_strain);
        void set_consistent_tangent_operator(const SymmetricTensor<4, dim> &consistent_tangent_operator);
        void set_principal_stresses(const std::vector<double> &principal_stresses);
        void set_accumulated_plastic_strain(double &accumulated_plastic_strain);
        void set_is_plastic(bool is_plastic);

        // NOTE: not all the variables which are stored by the point history are being accessed in the code
        SymmetricTensor<2, dim> get_stress() const;
        SymmetricTensor<2, dim> get_strain() const;
        SymmetricTensor<4, dim> get_consistent_tangent_operator() const;
        double get_accumulated_plastic_strain() const;
        bool get_is_plastic();

        // The following functions are used to get a few of the state variables from the previous time-step for the
        // return mapping algorithm. This can be seen in Box 7.3 (for Von-Mises) and Box 8.1 (for Tresca) in
        // the textbook.
        double get_stored_accumulated_plastic_strain() const;
        SymmetricTensor<2, dim> get_stored_elastic_strain() const;

    private:
        SymmetricTensor<2, dim> stress;
        SymmetricTensor<2, dim> elastic_strain;
        SymmetricTensor<2, dim> plastic_strain;
        SymmetricTensor<2, dim> strain;
        SymmetricTensor<2, dim> back_stress;
        SymmetricTensor<4, dim> consistent_tangent_operator;
        std::vector<double> principal_stresses;
        double accumulated_plastic_strain;
        bool is_plastic;

        SymmetricTensor<2, dim> stored_stress;
        SymmetricTensor<2, dim> stored_elastic_strain;
        SymmetricTensor<2, dim> stored_plastic_strain;
        SymmetricTensor<2, dim> stored_strain;
        SymmetricTensor<2, dim> stored_back_stress;
        SymmetricTensor<4, dim> stored_consistent_tangent_operator;
        std::vector<double> stored_principal_stresses;
        double stored_accumulated_plastic_strain;
        bool stored_is_plastic;
    };

    template <int dim>
    PointHistory<dim>::PointHistory(double K, double G)
    {
        // setting the initial value of the consistent tangent operator to the elastic consistent tangent operator
        consistent_tangent_operator = 2.0 * G * (identity_tensor<dim>() - 1.0 / 3.0 *
                outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())) + K *
                        outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());

        is_plastic = false;

        stored_is_plastic = false;
    }


    // Setter functions
    template <int dim>
    void PointHistory<dim>::set_stress(const SymmetricTensor<2, dim> &stress)
    {
        this->stress = stress;
    }

    template <int dim>
    void PointHistory<dim>::set_elastic_strain(const SymmetricTensor<2, dim> &elastic_strain)
    {
        this->elastic_strain = elastic_strain;
    }

    template <int dim>
    void PointHistory<dim>::set_strain(const SymmetricTensor<2, dim> &strain)
    {
        this->strain = strain;
    }

    template <int dim>
    void PointHistory<dim>::set_principal_stresses(const std::vector<double> &principal_stresses)
    {
        this->principal_stresses = principal_stresses;
    }

    template <int dim>
    void PointHistory<dim>::set_consistent_tangent_operator(const SymmetricTensor<4, dim> &consistent_tangent_operator)
    {
        this->consistent_tangent_operator = consistent_tangent_operator;
    }

    template <int dim>
    void PointHistory<dim>::set_accumulated_plastic_strain(double &accumulated_plastic_strain)
    {
        if (accumulated_plastic_strain >= stored_accumulated_plastic_strain)
            this->accumulated_plastic_strain = accumulated_plastic_strain;
    }

    template <int dim>
    void PointHistory<dim>::set_is_plastic(bool is_plastic)
    {
        this->is_plastic = is_plastic;
    }

    // Getter functions
    template <int dim>
    SymmetricTensor<2, dim> PointHistory<dim>::get_stress() const
    {
        return stress;
    }

    template<int dim>
    SymmetricTensor<2, dim> PointHistory<dim>::get_strain() const
    {
        return strain;
    }

    template <int dim>
    SymmetricTensor<4, dim> PointHistory<dim>::get_consistent_tangent_operator() const
    {
        return consistent_tangent_operator;
    }

    template<int dim>
    double PointHistory<dim>::get_accumulated_plastic_strain() const
    {
        return accumulated_plastic_strain;
    }

    template<int dim>
    bool PointHistory<dim>::get_is_plastic()
    {
        return is_plastic;
    }

    template <int dim>
    void PointHistory<dim>::store_internal_variables()
    {
        stored_stress = stress;
        stored_elastic_strain = elastic_strain;
        stored_plastic_strain = plastic_strain;
        stored_strain = strain;
        stored_back_stress = back_stress;
        stored_consistent_tangent_operator = consistent_tangent_operator;
        stored_principal_stresses = principal_stresses;
        stored_accumulated_plastic_strain = accumulated_plastic_strain;
        stored_is_plastic = is_plastic;
    }


    template<int dim>
    double PointHistory<dim>::get_stored_accumulated_plastic_strain() const
    {
        return stored_accumulated_plastic_strain;
    }

    template<int dim>
    SymmetricTensor<2, dim> PointHistory<dim>::get_stored_elastic_strain() const
    {
        return stored_elastic_strain;
    }


    template <int dim>
    class ConstitutiveLaw
    {
    public:
        ConstitutiveLaw(const double sigma_0,               // initial yield stress
                        const double hardening_slope,       // hardening slope (H in the textbook)
                        const double kappa,                 // bulk modulus (K in the textbook)
                        const double mu);                   // shear modulus (G in the textbook)

        // The following function performs the return-mapping algorithm (for the Tresca yield-criteria)
        // and determines the derivative of stress with respect to strain based off of whether a one-vector or a
        // two-vector return was used.
        void return_mapping_and_derivative_stress_strain(const SymmetricTensor<2, dim> &delta_strain,
                                                         std::shared_ptr<PointHistory<dim>> &qph,
                                                         std::string yield_criteria) const;

        SymmetricTensor<4, dim> derivative_of_isotropic_tensor(Tensor<1, dim> x, Tensor<2, dim> e,
                                                               SymmetricTensor<2, dim>, SymmetricTensor<2, dim> Y,
                                                               SymmetricTensor<2, dim> dy_dx) const;

    private:
        const double sigma_0;
        const double kappa;
        const double mu;
        const double hardening_slope;
    };


    template <int dim>
    ConstitutiveLaw<dim>::ConstitutiveLaw(const double sigma_0,
                                          const double hardening_slope,
                                          const double kappa,
                                          const double mu)

            : sigma_0(sigma_0)
            , kappa(kappa)
            , mu(mu)
            , hardening_slope(hardening_slope)
    {}


    // Function which computes and sorts the eigenvalues and eigenvectors of a tensor in descending order
    template <int dim>
    std::pair<Tensor<1, dim>, Tensor<2, dim>> compute_principal_values_vectors(const SymmetricTensor<2, dim> &A)
    {
        auto eigenvector_pairs = eigenvectors(A);

        std::sort(eigenvector_pairs.begin(), eigenvector_pairs.end(),
                  [](const std::pair<double, Tensor<1, dim>> &a, const std::pair<double, Tensor<1, dim>> &b) {
                      return a.first > b.first;
                  });

        Tensor<1, dim> sorted_eigenvalues;
        Tensor<2, dim> sorted_eigenvectors_tensor;

        for (unsigned int i = 0; i < dim; ++i)
        {
            sorted_eigenvalues[i] = eigenvector_pairs[i].first;
            for (unsigned int j = 0; j < dim; ++j)
            {
                sorted_eigenvectors_tensor[j][i] = eigenvector_pairs[i].second[j];
            }
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


    // Function which transfers a tensor from the principal directions to the original co-ordinate system
    template <int dim>
    SymmetricTensor<2, dim> reconstruct_tensor(const std::array<double, dim> &principal_values,
                                               const Tensor<2, dim> &eigenvectors_tensor)
    {
        Tensor<2, dim> principal_values_tensor;
        for (unsigned int i = 0; i < dim; ++i)
        {
            principal_values_tensor[i][i] = principal_values[i];
        }
        return symmetrize(eigenvectors_tensor * principal_values_tensor * transpose(eigenvectors_tensor));
    }

    // Function which computes the yield stress
    double yield_stress(double sigma_y_old , double H, double epsilon_p)
    {
        return sigma_y_old + H * epsilon_p;
    }

    template <int dim>
    void ConstitutiveLaw<dim>::return_mapping_and_derivative_stress_strain(
            const SymmetricTensor<2, dim> &delta_strain,
            std::shared_ptr<PointHistory<dim>> &qph,
            std::string yield_criteria) const
    {
        Assert(dim == 3, ExcNotImplemented());

        // Material properties
        const double yield_stress_0 = sigma_0;
        const double G = mu;
        const double K = kappa;
        const double H = hardening_slope;

        SymmetricTensor<2, dim> elastic_strain_trial = qph->get_stored_elastic_strain() + delta_strain;
        double accumulated_plastic_strain_n = qph->get_stored_accumulated_plastic_strain();

        // Elastic Predictor Step (Box 8.1, Step i) from the textbook
        auto elastic_strain_trial_eigenvector_pairs = compute_principal_values_vectors(elastic_strain_trial);
        auto elastic_strain_trial_eigenvalues = elastic_strain_trial_eigenvector_pairs.first;
        auto elastic_strain_trial_eigenvectors_matrix = elastic_strain_trial_eigenvector_pairs.second;

        double accumulated_plastic_strain_trial = accumulated_plastic_strain_n;
        SymmetricTensor<2, dim> deviatoric_stress_trial = 2.0 * G * deviator(elastic_strain_trial);
        double e_v_trial = trace(elastic_strain_trial);
        double p_trial = K * e_v_trial;

        SymmetricTensor<2, dim> ds_de;
        SymmetricTensor<2, dim> dsigma_de;

        // Initializing state variables at n+1
        SymmetricTensor<2, dim> elastic_strain_n1;
        SymmetricTensor<2, dim> deviatoric_stress_n1;
        double accumulated_plastic_strain_n1;
        SymmetricTensor<2, dim> s_n1;                  // deviatoric stress at n+1 in the principal directions
        SymmetricTensor<2, dim> stress_n1;

        // Spectral decomposition (Box 8.1, Step ii) from the textbook
        auto deviatoric_stress_trial_eigenvalues = compute_principal_values(deviatoric_stress_trial);

        double s1, s2, s3;        // declare principal deviatoric stresses

        // NOTE: Set this to be slightly more forgiving
        double tolerance = 1.e-5;  // tolerance for Newton iterations

        double delta_gamma;

        double q_trial;

        SymmetricTensor<4, dim> consistent_tangent_operator;

        if (yield_criteria == "Tresca")
        {
            // Plastic admissibility check (Box 8.1, Step iii) from the textbook
            double phi = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[dim - 1]
                         - yield_stress(yield_stress_0, H, accumulated_plastic_strain_trial);

            if (phi <= 0)
            {
                // Elastic step therefore setting values at n+1 to trial values
                elastic_strain_n1 = elastic_strain_trial;
                accumulated_plastic_strain_n1 = accumulated_plastic_strain_trial;

                s1 = deviatoric_stress_trial_eigenvalues[0];
                s2 = deviatoric_stress_trial_eigenvalues[1];
                s3 = deviatoric_stress_trial_eigenvalues[2];

                qph-> set_principal_stresses({s1, s2, s3});

                double p_n1 = p_trial;

                s_n1[0][0] = s1;
                s_n1[1][1] = s2;
                s_n1[2][2] = s3;

                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                // Stress has to be transformed back to the original co-ordinate system before storing
                SymmetricTensor<2, dim> stress_n1_reconstructed =
                        reconstruct_tensor({{stress_n1[0][0],stress_n1[1][1], stress_n1[2][2]}},
                                           elastic_strain_trial_eigenvectors_matrix);

                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                consistent_tangent_operator = 2.0 * G * (identity_tensor<dim>() -
                        1.0 / 3.0 * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())) +
                                K * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());

                qph->set_consistent_tangent_operator(consistent_tangent_operator);
            }
            else
            {
                // Plastic step: Return Mapping (Box 8.1, Step iv) from the textbook

                qph->set_is_plastic(true);

                double residual =
                        deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[dim - 1] -
                        yield_stress(yield_stress_0, H, accumulated_plastic_strain_n);

                bool local_newton_converged_one_v = false;

                // One-vector return to main plane (Box 8.2) from the textbook
                // TODO: make max inner steps an input
                for (unsigned int iteration = 0; iteration < 50; ++iteration) {
                    // Update yield stress after accumulating plastic strain (Box 8.2, Step ii)
                    // NOTE: for linear isotropic hardening; the hardening slope is constant
                    double residual_derivative = -4.0 * G - H;
                    delta_gamma -= residual / residual_derivative;

                    // Compute the residual (Box 8.2, Step ii)
                    residual = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[dim - 1] -
                               4.0 * G * delta_gamma - yield_stress(yield_stress_0, H,
                                                                    accumulated_plastic_strain_n + delta_gamma);

                    if (std::abs(residual) <= tolerance) {
                        // Update principal deviatoric stresses (Box 8.2, Step iii)
                        s1 = deviatoric_stress_trial_eigenvalues[0] - 2.0 * G * delta_gamma;
                        s2 = deviatoric_stress_trial_eigenvalues[1];
                        s3 = deviatoric_stress_trial_eigenvalues[2] + 2.0 * G * delta_gamma;

                        accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma;

                        local_newton_converged_one_v = true;

                        break;
                    }
                }
                // Checking if the Newton iteration convergstd::to_string(std::to_string(std::to_string(std::to_string(ed
                AssertThrow(local_newton_converged_one_v,
                            ExcMessage("Newton iteration did not converge for one-vector return with a residual of "+std::to_string(residual)));

                // Check if the updated principal stresses satisfy s1 >= s2 >= s3 (Box 8.1, Step iv.b) from the textbook
                if (s1 >= s2 && s2 >= s3) {
//                std::cout << "Plastic: one-vector return" << std::endl;

                    double f = (2.0 * G) / (4.0 * G + H);

                    // This relates to equation 8.4.1 in the textbook
                    ds_de[0][0] = 2.0 * G * (1.0 - f);
                    ds_de[0][2] = 2.0 * G * f;
                    ds_de[1][1] = 2.0 * G;
                    ds_de[2][0] = 2.0 * G * f;
                    ds_de[2][2] = 2.0 * G * (1.0 - f);
                }
                else
                {
//                   std::cout << "Plastic: two-vector return" << std::endl;

                    if (deviatoric_stress_trial_eigenvalues[0] + deviatoric_stress_trial_eigenvalues[2] -
                        (2.0 * deviatoric_stress_trial_eigenvalues[1]) > 0.0)
                    {
                        // Right corner return
//                        std::cout << "Right corner return" << std::endl;

                        // This relates to equation 8.52 in the textbook
                        double daa = -4.0 * G - H;
                        double dab = -2.0 * G - H;
                        double dba = -2.0 * G - H;
                        double dbb = -4.0 * G - H;

                        double det_d = daa * dbb - dab * dba;

                        ds_de[0][0] = 2.0 * G * (1.0 - ((8.0 * pow(G, 2.0)) / det_d));
                        ds_de[0][1] = ((4.0 * pow(G, 2.0)) / det_d) * (dab - daa);
                        ds_de[0][2] = ((4.0 * pow(G, 2.0)) / det_d) * (dba - dbb);
                        ds_de[1][0] = (8.0 * pow(G, 3.0)) / det_d;
                        ds_de[1][1] = 2.0 * G * (1.0 + ((2.0 * G * daa) / det_d));
                        ds_de[1][2] = -(4.0 * pow(G, 2.0) / det_d) * dba;
                        ds_de[2][0] = (8.0 * pow(G, 3.0)) / det_d;
                        ds_de[2][1] = -(4.0 * pow(G, 2.0) / det_d) * dab;
                        ds_de[2][2] = 2.0 * G * (1.0 + ((2.0 * G * dbb) / det_d));

                        Vector<double> delta_gamma_vector(2);
                        delta_gamma_vector = 0.0;

                        double s_b = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[1];
                        double s_a = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[2];

                        Vector<double> residual_vector(2);
                        residual_vector[0] = s_a - yield_stress(yield_stress_0, H,
                                                                accumulated_plastic_strain_n);
                        residual_vector[1] = s_b - yield_stress(yield_stress_0, H,
                                                                accumulated_plastic_strain_n);

                        Vector<double> delta_gamma_vector_update(2);
                        delta_gamma_vector_update = 0.0;

                        double delta_gamma_sum;

                        FullMatrix<double> d_matrix(2, 2);
                        d_matrix(0, 0) = -4.0 * G - H;
                        d_matrix(0, 1) = -2.0 * G - H;
                        d_matrix(1, 0) = -2.0 * G - H;
                        d_matrix(1, 1) = -4.0 * G - H;

                        FullMatrix<double> d_matrix_inverse(2, 2);
                        d_matrix_inverse.invert(d_matrix);

                        // Newton iteration for two-vector return (Box 8.3, Step ii) from the textbook
                        for (unsigned int iteration = 0; iteration < 50; ++iteration) {
                            delta_gamma_sum = delta_gamma_vector[0] + delta_gamma_vector[1];
                            accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma_sum;

                            residual_vector *= -1.0;
                            d_matrix_inverse.vmult(delta_gamma_vector_update, residual_vector);

                            delta_gamma_vector += delta_gamma_vector_update;

                            residual_vector[0] = s_a - 2.0 * G * (2.0 * delta_gamma_vector[0] + delta_gamma_vector[1]) -
                                                 yield_stress(yield_stress_0, H, accumulated_plastic_strain_n1);
                            residual_vector[1] = s_b - 2.0 * G * (delta_gamma_vector[0] + 2.0 * delta_gamma_vector[1]) -
                                                 yield_stress(yield_stress_0, H, accumulated_plastic_strain_n1);

                            if (abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance) {
                                s1 = deviatoric_stress_trial_eigenvalues[0] - 2.0 * G * (delta_gamma_vector[0] +
                                                                                         delta_gamma_vector[1]);
                                s2 = deviatoric_stress_trial_eigenvalues[1] + 2.0 * G * delta_gamma_vector[1];
                                s3 = deviatoric_stress_trial_eigenvalues[2] + 2.0 * G * delta_gamma_vector[0];

                                break;
                            }
                        }
                        AssertThrow(abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance,
                                    ExcMessage("Two-vector return did not converge"));
                    }
                    else
                    {
                        // Left corner return
//                       std::cout << "Left corner return" << std::endl;

                        double daa = -4.0 * G - H;
                        double dab = -2.0 * G - H;
                        double dba = -2.0 * G - H;
                        double dbb = -4.0 * G - H;

                        double det_d = daa * dbb - dab * dba;

                        ds_de[0][0] = 2.0 * G * (1.0 + ((2.0 * G * dbb) / det_d));
                        ds_de[0][1] = -(4.0 * pow(G, 2.0) / det_d) * dab;
                        ds_de[0][2] = (8.0 * pow(G, 3.0)) / det_d;
                        ds_de[1][0] = -(4.0 * pow(G, 2.0) / det_d) * dba;
                        ds_de[1][1] = 2.0 * G * (1.0 + ((2.0 * G * daa) / det_d));
                        ds_de[1][2] = (8.0 * pow(G, 3.0)) / det_d;
                        ds_de[2][0] = (4.0 * pow(G, 2.0) / det_d) * (dba - dbb);
                        ds_de[2][1] = (4.0 * pow(G, 2.0) / det_d) * (dab - daa);
                        ds_de[2][2] = 2.0 * G * (1.0 - ((8.0 * pow(G, 2.0)) / det_d));

                        Vector<double> delta_gamma_vector(2);
                        delta_gamma_vector = 0.0;

                        double s_b = deviatoric_stress_trial_eigenvalues[1] - deviatoric_stress_trial_eigenvalues[2];
                        double s_a = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[2];

                        Vector<double> residual_vector(2);
                        residual_vector[0] = s_a - yield_stress(yield_stress_0, H,
                                                                accumulated_plastic_strain_n);
                        residual_vector[1] = s_b - yield_stress(yield_stress_0, H,
                                                                accumulated_plastic_strain_n);

                        Vector<double> delta_gamma_vector_update(2);
                        delta_gamma_vector_update = 0.0;

                        double delta_gamma_sum;

                        FullMatrix<double> d_matrix(2, 2);
                        d_matrix(0, 0) = -4.0 * G - H;
                        d_matrix(0, 1) = -2.0 * G - H;
                        d_matrix(1, 0) = -2.0 * G - H;
                        d_matrix(1, 1) = -4.0 * G - H;

                        FullMatrix<double> d_matrix_inverse(2, 2);
                        d_matrix_inverse.invert(d_matrix);

                        // Newton iteration for two-vector return (Box 8.3, Step ii) from the textbook
                        for (unsigned int iteration = 0; iteration < 50; ++iteration) {
                            delta_gamma_sum = delta_gamma_vector[0] + delta_gamma_vector[1];
                            accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma_sum;

                            residual_vector *= -1.0;
                            d_matrix_inverse.vmult(delta_gamma_vector_update, residual_vector);

                            delta_gamma_vector += delta_gamma_vector_update;

                            residual_vector[0] = s_a - 2.0 * G * (2.0 * delta_gamma_vector[0] + delta_gamma_vector[1]) -
                                                 yield_stress(yield_stress_0, H,
                                                              accumulated_plastic_strain_n1);
                            residual_vector[1] = s_b - 2.0 * G * (delta_gamma_vector[0] + 2.0 * delta_gamma_vector[1]) -
                                                 yield_stress(yield_stress_0, H,
                                                              accumulated_plastic_strain_n1);

                            if (abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance) {
                                s1 = deviatoric_stress_trial_eigenvalues[0] - 2.0 * G * delta_gamma_vector[0];
                                s2 = deviatoric_stress_trial_eigenvalues[1] - 2.0 * G * delta_gamma_vector[1];
                                s3 = deviatoric_stress_trial_eigenvalues[2] + 2.0 * G * (delta_gamma_vector[0] +
                                                                                         delta_gamma_vector[1]);

                                break;
                            }
                        }
                        AssertThrow(abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance,
                                    ExcMessage("Two-vector return did not converge"));
                    }
                }
                // Equation 8.46 from textbook
                for (unsigned int i = 0; i < 3; ++i)
                    for (unsigned int j = 0; j < 3; ++j) {
                        dsigma_de[i][j] = K;

                        for (unsigned int k = 0; k < 3; ++k) {
                            double delta_kj = (k == j) ? 1.0 : 0.0;
                            dsigma_de[i][j] += ds_de[i][k] * (delta_kj - 1.0 / 3.0);
                        }
                    }

                qph->set_principal_stresses({s1, s2, s3});

                double p_n1 = p_trial;

                s_n1[0][0] = s1;
                s_n1[1][1] = s2;
                s_n1[2][2] = s3;

                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                SymmetricTensor<2, dim> stress_n1_reconstructed =
                        reconstruct_tensor({{stress_n1[0][0], stress_n1[1][1], stress_n1[2][2]}},
                                           elastic_strain_trial_eigenvectors_matrix);

                // elastic strain has to be reconstructed back to the original co-ordinate system before storing
                elastic_strain_n1 = 1.0 / (2.0 * G) *
                                    reconstruct_tensor({{s_n1[0][0], s_n1[1][1], s_n1[2][2]}},
                                                       elastic_strain_trial_eigenvectors_matrix) +
                                    (1.0 / 3.0) * e_v_trial * unit_symmetric_tensor<dim>();

                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                consistent_tangent_operator =
                        derivative_of_isotropic_tensor(elastic_strain_trial_eigenvalues,
                                                       elastic_strain_trial_eigenvectors_matrix,
                                                       elastic_strain_trial, stress_n1, dsigma_de);

                qph->set_consistent_tangent_operator(consistent_tangent_operator);
            }
        }
        if (yield_criteria == "Von-Mises")
        {
            q_trial = sqrt((3.0 / 2.0) * scalar_product(deviatoric_stress_trial, deviatoric_stress_trial));

            if (q_trial - yield_stress(yield_stress_0, H, accumulated_plastic_strain_trial) <= 0)
            {
//                std::cout << "Elastic step" << std::endl;

                // Elastic step therefore setting values at n+1 to trial values
                elastic_strain_n1 = elastic_strain_trial;
                accumulated_plastic_strain_n1 = accumulated_plastic_strain_trial;

                double p_n1 = p_trial;

                s1 = deviatoric_stress_trial_eigenvalues[0];
                s2 = deviatoric_stress_trial_eigenvalues[1];
                s3 = deviatoric_stress_trial_eigenvalues[2];

                s_n1[0][0] = s1;
                s_n1[1][1] = s2;
                s_n1[2][2] = s3;

                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                SymmetricTensor<2, dim> stress_n1_reconstructed =
                        reconstruct_tensor({{stress_n1[0][0], stress_n1[1][1], stress_n1[2][2]}},
                                           elastic_strain_trial_eigenvectors_matrix);

                qph->set_principal_stresses({s1, s2, s3});
                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                consistent_tangent_operator = 2.0 * G * (identity_tensor<dim>() -
                        (1.0 / 3.0) * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())) +
                                              K * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());

                qph->set_consistent_tangent_operator(consistent_tangent_operator);
                qph->set_is_plastic(false);
            }
            else
            {
                qph->set_is_plastic(true);

//                std::cout << "Plastic step" << std::endl;

                double residual = q_trial - yield_stress(yield_stress_0, H, accumulated_plastic_strain_n);

                for (unsigned int iteration = 0; iteration < 100; ++iteration) {
                    const double residual_derivative = -3.0 * G - H;
                    delta_gamma -= residual / residual_derivative;

                    residual = q_trial - 3.0 * G * delta_gamma -
                               yield_stress(yield_stress_0, H, accumulated_plastic_strain_n + delta_gamma);

                    if (std::abs(residual) < tolerance) {
                        break;
                    }
                }
                AssertThrow(std::abs(residual) < tolerance,
                            ExcMessage("Newton iteration did not converge for Von-Mises yield criteria"));

                accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma;

                s_n1[0][0] = (1.0 - (delta_gamma * 3.0 * G) / q_trial) * deviatoric_stress_trial_eigenvalues[0];
                s_n1[1][1] = (1.0 - (delta_gamma * 3.0 * G) / q_trial) * deviatoric_stress_trial_eigenvalues[1];
                s_n1[2][2] = (1.0 - (delta_gamma * 3.0 * G) / q_trial) * deviatoric_stress_trial_eigenvalues[2];

                qph->set_principal_stresses({s_n1[0][0], s_n1[1][1], s_n1[2][2]});

                double p_n1 = p_trial;

                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                SymmetricTensor<2, dim> stress_n1_reconstructed =
                        reconstruct_tensor({{stress_n1[0][0], stress_n1[1][1], stress_n1[2][2]}},
                                           elastic_strain_trial_eigenvectors_matrix);

                elastic_strain_n1 = (1.0 / (2.0 * G)) *
                                    reconstruct_tensor({{s_n1[0][0], s_n1[1][1], s_n1[2][2]}},
                                                       elastic_strain_trial_eigenvectors_matrix) +
                                    (1.0 / 3.0) * e_v_trial * unit_symmetric_tensor<dim>();

                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                SymmetricTensor<2, dim> N_n1 = deviatoric_stress_trial / deviatoric_stress_trial.norm();

                consistent_tangent_operator = 2.0 * G * (1.0 - (delta_gamma * 3.0 * G) / q_trial) *
                                              (identity_tensor<dim>() -
                                               (1.0 / 3.0) * outer_product(unit_symmetric_tensor<dim>(),
                                                                           unit_symmetric_tensor<dim>())) +
                                              6.0 * G * G *
                                              (delta_gamma / q_trial - (1.0 / (3.0 * G + H))) *
                                              outer_product(N_n1, N_n1) +
                                              K *
                                              outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());

                qph->set_consistent_tangent_operator(consistent_tangent_operator);
            }
        }
    }

    template <int dim>
    SymmetricTensor<4, dim> symmetrize_tensor(const Tensor<4, dim> &D, const double rel_tol = 1e-3, bool check_for_symm = false)
    {
        SymmetricTensor<4, dim> symmetrized_D;

        // Loop through all components and symmetrize
        for (unsigned int i = 0; i < dim; ++i)
        {
            for (unsigned int j = 0; j < dim; ++j)
            {
                for (unsigned int k = 0; k < dim; ++k)
                {
                    for (unsigned int l = 0; l < dim; ++l)
                    {
                        // Symmetrize the tensor
                        symmetrized_D[i][j][k][l] = 1.0/3 * (    D[i][j][k][l]
                                                                 + D[j][i][k][l]
                                                                 + D[i][j][l][k]);
                    }
                }
            }
        }

        // Now check if the symmetrized tensor and original tensor are equivalent within tolerance
        if(check_for_symm)
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                    for (unsigned int k = 0; k < dim; ++k)
                        for (unsigned int l = 0; l < dim; ++l)
                        {
                            double original_value = D[i][j][k][l];
                            double symmetrized_value = symmetrized_D[i][j][k][l];
                            AssertThrow(std::fabs(original_value - symmetrized_value) <= abs(original_value)/rel_tol,
                                        ExcMessage("Tensor components are not symmetric within the specified tolerance."));
                        }

        return symmetrized_D;
    }

    // TODO: Check if the following function is correct
    template <int dim>
    SymmetricTensor<4, dim> ConstitutiveLaw<dim>::derivative_of_isotropic_tensor(
            Tensor<1, dim> x, Tensor<2, dim> e, SymmetricTensor<2, dim> X, SymmetricTensor<2, dim> Y,
            SymmetricTensor<2, dim> dy_dx) const
    {
        // For this model the following should be passed as the arguments of this function
        //   Y - stress at n+1 as a tensor
        //   X - trial elastic strain at n+1 as a tensor
        //   x - the eigenvalues of the trial elastic strain tensor
        //   e - the eigenvectors of the trial elastic strain tensor
        //   dy_dx - derivative of the stress (at n+1) with respect to the trial elastic strain (at n+1)

        // The eigenvalues of the stress at n+1 are not being computed in this function because the stress at n+1 is
        // being passed as a tensor in its principal directions therefore the principal stresses are on the diagonal
        // of the tensor.
        Vector<double> y(dim);
        y[0] = Y[0][0];
        y[1] = Y[1][1];
        y[2] = Y[2][2];

        // Calculate the 4th-order tensor d[X^2]/dX_ijkl as per equation A.46 in the textbook.
        SymmetricTensor<4, dim> dX2_dX;

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
                                delta_kj * X[i][l] );
                    }
                }
            }
        }

        // Compute the projection tensors Ei = ei ⊗ ei for each eigenvector
        std::array<Tensor<2, dim>, 3> E;

        // Initialize the 4th-order tensor D to zero
        Tensor<4, dim> D;

        // The following can be found in box A.6 of the textbook
        if (x[0] != x[1] && x[1] != x[2])
        {
            for (unsigned int a = 0; a < dim; ++a)
            {
                unsigned int b;
                unsigned int c;

                if (a == 0)
                {
                    b = 1;
                    c = 2;
                }
                if (a == 1)
                {
                    b = 2;
                    c = 0;
                }
                if (a == 2)
                {
                    b = 0;
                    c = 1;
                }

                // Compute the projection tensors Ei = ei ⊗ ei for each eigenvector
                E[a] = symmetrize(outer_product(e[a], e[a]));
                E[b] = symmetrize(outer_product(e[b], e[b]));
                E[c] = symmetrize(outer_product(e[c], e[c]));

                D += (y[a] / ((x[a] - x[b]) * (x[a] - x[c]))) * (dX2_dX - (x[b] + x[c]) * identity_tensor<dim>() -
                        ((x[a] - x[b]) + (x[a] - x[c])) * outer_product(E[a], E[a]) - (x[b] - x[c]) *
                        (outer_product(E[b], E[b]) - outer_product(E[c], E[c])));
            }
            for (unsigned int i = 0; i < dim; ++i)
            {
                for (unsigned int j = 0; j < dim; ++j)
                {
                    D += dy_dx[i][j] * outer_product(E[i], E[j]);
                }
            }
        }
        else if (x[0] == x[1] && x[1] == x[2])
        {
            D = (dy_dx[0][0] - dy_dx[0][1]) * identity_tensor<dim>() +
                dy_dx[0][1] * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
        }
        else
        {
            unsigned int a;
            unsigned int b;
            unsigned int c;

            if (x[0] == x[1] && x[1] != x[2])
            {
                a = 2;
                b = 0;
                c = 1;
            }
            else if (x[1] == x[2] && x[0] != x[1])
            {
                a = 0;
                b = 1;
                c = 2;
            }
            else
                AssertThrow(false, ExcMessage("Eigenvalues are not sorted in isotropic tensor derivative."));

            // The following are not deviatoric stress values
            double s1 = (y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c])) +
                    (1.0 / (x[a] - x[c])) * (dy_dx[c][b] - dy_dx[c][c]);

            double s2 = 2.0 * x[c] * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]))) +
                    ((x[a] + x[c]) / (x[a] - x[c])) * (dy_dx[c][b] - dy_dx[c][c]);

            double s3 = 2.0 * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c]))) +
                    (1.0 / ((x[a] - x[c]) * (x[a] - x[c]))) * (dy_dx[a][c] + dy_dx[c][a] - dy_dx[a][a] - dy_dx[c][c]);

            double s4 = 2.0 * x[c] * (y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c])) +
                    (1.0 / (x[a] - x[c])) * (dy_dx[a][c] - dy_dx[c][b]) + (x[c] / ((x[a] - x[c]) * (x[a] - x[c]))) *
                    (dy_dx[a][c] + dy_dx[c][a] - dy_dx[a][a] - dy_dx[c][c]);

            double s5 = 2.0 * x[c] * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c]))) +
                    (1.0 / (x[a] - x[c])) * (dy_dx[c][a] - dy_dx[c][b]) + (x[c] / ((x[a] - x[c]) * (x[a] - x[c]))) *
                    (dy_dx[a][c] + dy_dx[c][a] - dy_dx[a][a] - dy_dx[c][c]);

            double s6 = 2.0 * (x[c] * x[c]) * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c]))) +
                    ((x[a] * x[c]) / ((x[a] - x[c]) * (x[a] - x[c]))) *
                    (dy_dx[a][c] + dy_dx[c][a])  - ((x[c] * x[c]) / ((x[a] - x[c]) * (x[a] - x[c]))) *
                    (dy_dx[a][a] + dy_dx[c][c]) - ((x[a] + x[c]) / (x[a] - x[c])) * dy_dx[c][b];

            D = s1 * dX2_dX - s2 * identity_tensor<dim>() - s3 * outer_product(X, X) +
                    s4 * outer_product(X, unit_symmetric_tensor<dim>()) +
                    s5 * outer_product(unit_symmetric_tensor<dim>(), X) -
                    s6 * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
        }

        SymmetricTensor<4, dim> D_sym = symmetrize_tensor(D);

        return D_sym;
    }


    namespace EquationData
    {
        template<int dim>
        class BoundaryForce : public TensorFunction<1, dim, double>{
        public:
            BoundaryForce(std::string problem_type, double inner_radius, double outer_radius, double normal_force,
                          double torque) : TensorFunction<1, dim, double>(), problem_type(problem_type),
                          inner_radius(inner_radius), outer_radius(outer_radius), normal_force(normal_force),
                          torque(torque){}

            virtual Tensor<1, dim> value(const Point<dim> &p) const override
            {
                Tensor<1, dim> traction;

                if (problem_type == "torsion")
                {
                    const double J = (M_PI / 2.0) * (std::pow(outer_radius, 4) - std::pow(inner_radius, 4));

                    traction[0] = (-torque * p[1]) / J;
                    traction[1] = (torque * p[0]) / J;
//                    traction[2] = normal_force / (M_PI * (std::pow(outer_radius, 2) - std::pow(inner_radius, 2)));

                    // NOTE: the following ensures that the axial stress and shear stress are equal
                    //  this is used to compare to the analytical solution
                    const double N = ((2 * std::sqrt(std::pow(p[0], 2) + std::pow(p[1], 2))) /
                            (std::pow(outer_radius, 2) + std::pow(inner_radius, 2))) * torque;

                    traction[2] = N / (M_PI * (std::pow(outer_radius, 2) - std::pow(inner_radius, 2)));
                }
                else if (problem_type == "tensile")
                {
                    traction[2] = 412.305;
                }
                else if (problem_type == "shear")
                {
                    traction[0] = 234.531;
                }

                return traction;
            }

        private:
            const std::string problem_type;
            const double inner_radius;
            const double outer_radius;
            const double normal_force;
            const double torque;
        };
    }


    template <int dim>
    class PlasticityProblem
    {
    public:
        PlasticityProblem(const ParameterHandler& prm);

        void run();

        static void declare_parameters(ParameterHandler& prm);

        const std::string loading;

    private:
        void make_grid(std::string mesh);
        void setup_system();
        void compute_dirichlet_constraints(const double displacement, std::string problem_type);
        void assemble_newton_system(const TrilinosWrappers::MPI::Vector &solution,
                                    const TrilinosWrappers::MPI::Vector &old_solution, const bool rhs_only = false,
                                    unsigned int n_t_steps = 1, unsigned int t_step = 0);
        void solve_newton_system(TrilinosWrappers::MPI::Vector &newton_increment);
        void solve_newton(unsigned int n_t_steps = 1, unsigned int t_step = 0);
        void move_mesh(const TrilinosWrappers::MPI::Vector& displacement) const;
        void output_results(const unsigned int current_time_step, const std::string output_name = "solution");

        void store_internal_variables();

        MPI_Comm mpi_communicator;
        ConditionalOStream pcout;
        TimerOutput computing_timer;

        const unsigned int n_initial_global_refinements;
        parallel::distributed::Triangulation<dim> triangulation;

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

        IndexSet active_set;

        Vector<double> accumulated_plastic_strain;

        std::vector<SymmetricTensor<2, dim>> stress_at_q_points;

        TrilinosWrappers::SparseMatrix newton_matrix;

        TrilinosWrappers::MPI::Vector solution;
        TrilinosWrappers::MPI::Vector old_solution;
        TrilinosWrappers::MPI::Vector delta_solution;
        TrilinosWrappers::MPI::Vector newton_rhs;
        TrilinosWrappers::MPI::Vector newton_rhs_uncondensed;
        TrilinosWrappers::MPI::Vector diag_mass_matrix_vector;
        TrilinosWrappers::MPI::Vector stress_tensor_diagonal;
        TrilinosWrappers::MPI::Vector stress_tensor_off_diagonal;
        TrilinosWrappers::MPI::Vector stress_tensor_tmp;
        TrilinosWrappers::MPI::Vector strain_tensor_tmp;
        TrilinosWrappers::MPI::Vector accumulated_plastic_strain_vector;
        TrilinosWrappers::MPI::Vector is_plastic_vector;

        const double sigma_0, hardening_slope, kappa, mu;
        ConstitutiveLaw<dim> constitutive_law;

        const std::string base_mesh;

        const std::string problem;

        const double applied_displacement;

        const std::string yield_criteria;

        struct RefinementStrategy
        {
            enum value
            {
                refine_global,
                refine_percentage,
                refine_fix_dofs
            };
        };

        // typename RefinementStrategy::value refinement_strategy;

        const bool transfer_solution;
        std::string output_dir;
        const unsigned int n_time_steps;
        unsigned int current_step;

        const double outer_radius;
        const double inner_radius;

        const double normal_force;
        const double torque;
    };


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
                "number of time-steps",
                "5",
                Patterns::Integer(),
                "Number of psuedo time-steps for the simulation.");
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
        prm.declare_entry(
                "base mesh",
                "box",
                Patterns::Selection("box|cylinder"),
                "Select the shape of the domain: 'box' or hollow 'cylinder'.");
        prm.declare_entry(
                "problem",
                "tensile",
                Patterns::Selection("tensile|shear|torsion"),
                "Select the problem type: uniaxial 'tension', simple 'shear', 'torsion' and tension of cylinder.");
        prm.declare_entry(
                "loading",
                "no reverse",
                Patterns::Selection("reverse|no reverse"),
                "Select the loading type: 'reverse' is for reversed loading and 'no reverse' is for one-directional loading.");
        prm.declare_entry(
                "applied displacement",
                "-0.002",
                Patterns::Double(),
                "Applied displacement to the top of the box");
        prm.declare_entry(
                "yield-criteria",
                "Von-Mises",
                Patterns::Selection("Von-Mises|Tresca"),
                "Select the yield-criteria: 'Von-Mises' or 'Tresca'.");
        prm.declare_entry(
                "outer radius",
                "10",
                Patterns::Double(),
                "The outer radius of the hollow cylinder.");
        prm.declare_entry(
                "inner radius",
                "9",
                Patterns::Double(),
                "The inner radius of the hollow cylinder.");
        prm.declare_entry(
                "normal force",
                "100000.0",
                Patterns::Double(),
                "The normal force applied the free-end.");
        prm.declare_entry(
                "torque",
                "1000.0",
                Patterns::Double(),
                "The torque applied the free-end.");
    }


    template <int dim>
    PlasticityProblem<dim>::PlasticityProblem(const ParameterHandler& prm)
            : mpi_communicator(MPI_COMM_WORLD)
            , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)

            , computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::never,
                              TimerOutput::wall_times)

            , n_initial_global_refinements(prm.get_integer("number of initial refinements"))
            , triangulation(mpi_communicator)
            , fe_degree(prm.get_integer("polynomial degree"))
            , fe(FE_Q<dim>(QGaussLobatto<1>(fe_degree + 1)) ^ dim)
            , fe_scalar(fe_degree)
            , dof_handler(triangulation)
            , dof_handler_scalar(triangulation)

            , hardening_slope(1550.0)
            , kappa(166670.0)
            , mu(76920.0)
            , sigma_0(400.0)
            , constitutive_law(sigma_0, hardening_slope, kappa, mu)

            , base_mesh(prm.get("base mesh"))

            , problem(prm.get("problem"))

            , loading(prm.get("loading"))

            , applied_displacement(prm.get_double("applied displacement"))

            , yield_criteria(prm.get("yield-criteria"))

            , transfer_solution(prm.get_bool("transfer solution"))
            , n_time_steps(prm.get_integer("number of time-steps"))
            , current_step(0)

            , outer_radius(prm.get_double("outer radius"))
            , inner_radius(prm.get_double("inner radius"))

            , normal_force(prm.get_double("normal force"))
            , torque(prm.get_double("torque"))
    {
//        std::string strat = prm.get("refinement strategy");
//        if (strat == "global")
//            refinement_strategy = RefinementStrategy::refine_global;
//        else if (strat == "percentage")
//            refinement_strategy = RefinementStrategy::refine_percentage;
//        else
//            AssertThrow(false, ExcNotImplemented());

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


    template <int dim>
    void PlasticityProblem<dim>::make_grid(std::string mesh)
    {
        if (mesh == "box")
        {
            // unit cube mesh
            const Point<dim> p1(0., 0., 0.);
            const Point<dim> p2(1.0, 1.0, 1.0);

            GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
        }
        else if (mesh == "cylinder")
        {
            // Parameters for the hollow cylinder
            const double length = 1.0;               // Height of the cylinder
            const unsigned int n_radial_cells = 10;  // Number of radial cells
            const unsigned int n_axial_cells = 5;    // Number of axial cells
            const bool colorize = true;              // Assign boundary IDs

            GridGenerator::cylinder_shell(
                    triangulation,
                    length,
                    inner_radius,
                    outer_radius,
                    n_radial_cells,
                    n_axial_cells,
                    colorize);

            // Assign boundary IDs for lines x=0 and y=0 on the face z=0
            for (auto &cell : triangulation.active_cell_iterators())
            {
                for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; ++face)
                {
                    if (cell->face(face)->at_boundary())
                    {
                        const auto center = cell->face(face)->center();

                        // Check if the face lies on z=0
                        if (std::abs(center[2]) < 1e-12)
                        {
                            // Loop through the vertices of the face to check conditions
                            for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_face; ++v)
                            {
                                const auto vertex = cell->face(face)->vertex(v);

                                // Check if x=0
                                if (std::abs(vertex[0]) < 1e-12)
                                {
                                    cell->face(face)->set_boundary_id(111); // Boundary ID 1 for x=0
                                }

                                // Check if y=0
                                if (std::abs(vertex[1]) < 1e-12)
                                {
                                    cell->face(face)->set_boundary_id(222); // Boundary ID 2 for y=0
                                }
                            }
                        }
                    }
                }
            }
        }
        triangulation.refine_global(n_initial_global_refinements);  // refine the mesh globally
    }


    template <int dim>
    void PlasticityProblem<dim>::setup_system()
    {
        pcout << "Setting up system..." << std::endl;

        {
            TimerOutput::Scope t(computing_timer, "Setup: distribute DoFs");
            dof_handler.distribute_dofs(fe);
            dof_handler_scalar.distribute_dofs(fe_scalar);

            locally_owned_dofs = dof_handler.locally_owned_dofs();
            locally_owned_scalar_dofs = dof_handler_scalar.locally_owned_dofs();
            locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
            locally_relevant_scalar_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_scalar);
        }
        {
            TimerOutput::Scope t(computing_timer, "Setup: vectors");
            solution.reinit(locally_relevant_dofs, mpi_communicator);
            old_solution.reinit(locally_owned_dofs, mpi_communicator);
            delta_solution.reinit(locally_owned_dofs, mpi_communicator);
            newton_rhs.reinit(locally_owned_dofs, mpi_communicator);
            newton_rhs_uncondensed.reinit(locally_owned_dofs, mpi_communicator);
            diag_mass_matrix_vector.reinit(locally_owned_dofs, mpi_communicator);
            accumulated_plastic_strain.reinit(triangulation.n_active_cells());
            stress_tensor_diagonal.reinit(locally_owned_dofs, mpi_communicator);
            stress_tensor_off_diagonal.reinit(locally_owned_dofs, mpi_communicator);
            stress_tensor_tmp.reinit(locally_owned_scalar_dofs, mpi_communicator);
            strain_tensor_tmp.reinit(locally_owned_scalar_dofs, mpi_communicator);
        }
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

        const QGauss<dim> quadrature_formula(fe_degree + 1);

        quadrature_point_history.resize(triangulation.n_locally_owned_active_cells() * quadrature_formula.size());

        unsigned int local_history_index = 0;
        for (const auto &cell : triangulation.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                {
                    quadrature_point_history[local_history_index] = std::make_shared<PointHistory<dim>>(kappa, mu);
                    ++local_history_index;
                }
            }
        }
    }


    template <int dim>
    void PlasticityProblem<dim>::compute_dirichlet_constraints(const double displacement, std::string problem_type)
    {
        constraints_dirichlet_and_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
        constraints_dirichlet_and_hanging_nodes.merge(constraints_hanging_nodes);

        const FEValuesExtractors::Scalar x_displacement(0);
        const FEValuesExtractors::Scalar y_displacement(1);
        const FEValuesExtractors::Scalar z_displacement(2);

        if (problem_type == "tensile")
        {
            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    // top face
                    5,
                    Functions::ConstantFunction<dim>(displacement, dim),
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
                    // left face
                    0,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(x_displacement));

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    // back face
                    2,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(y_displacement));
        }
        else if (problem_type == "shear")
        {
            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    Functions::ConstantFunction<dim>(displacement, dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(x_displacement));

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    (fe.component_mask(y_displacement) | fe.component_mask(z_displacement)));

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    4,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    (fe.component_mask(x_displacement) | fe.component_mask(z_displacement) | fe.component_mask(y_displacement)));

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    2,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(y_displacement));

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    3,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(y_displacement));

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    0,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(z_displacement));

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    1,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(z_displacement));
        }
        else if (problem_type == "torsion")
        {
            // the following will fix the bottom in all directions
            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    2,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    (fe.component_mask(x_displacement) | fe.component_mask(z_displacement) | fe.component_mask(y_displacement)));


            // the following is to fix lines on the base of the cylinder
//            VectorTools::interpolate_boundary_values(
//                    dof_handler,
//                    2,
//                    Functions::ZeroFunction<dim>(dim),
//                    constraints_dirichlet_and_hanging_nodes,
//                    fe.component_mask(z_displacement));
//
//            VectorTools::interpolate_boundary_values(
//                    dof_handler,
//                    111,
//                    Functions::ZeroFunction<dim>(dim),
//                    constraints_dirichlet_and_hanging_nodes,
//                    (fe.component_mask(y_displacement) | fe.component_mask(z_displacement)));
//
//            VectorTools::interpolate_boundary_values(
//                    dof_handler,
//                    222,
//                    Functions::ZeroFunction<dim>(dim),
//                    constraints_dirichlet_and_hanging_nodes,
//                    (fe.component_mask(x_displacement) | fe.component_mask(z_displacement)));


            // the following is to apply a Dirichlet BC to the top face
//            VectorTools::interpolate_boundary_values(
//                    dof_handler,
//                    3,
//                    Functions::ConstantFunction<dim>(displacement, dim),
//                    constraints_dirichlet_and_hanging_nodes,
//                    fe.component_mask(z_displacement));
        }
    }


    template <int dim>
    void PlasticityProblem<dim>::assemble_newton_system(const TrilinosWrappers::MPI::Vector &solution,
                                                        const TrilinosWrappers::MPI::Vector &old_solution,
                                                        const bool rhs_only, unsigned int n_t_steps, unsigned int t_step)
    {
        TimerOutput::Scope t(computing_timer, "Assembling");

        const QGauss<dim> quadrature_formula(fe_degree + 1);
        const QGauss<dim - 1> face_quadrature_formula(fe_degree + 1);

        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients | update_JxW_values);


        FEFaceValues<dim> fe_values_face(fe,
                                         face_quadrature_formula,
                                         update_values | update_quadrature_points |
                                         update_JxW_values);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const FEValuesExtractors::Vector displacement(0);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                cell_matrix = 0.;
                cell_rhs = 0.;

                std::vector<SymmetricTensor<2, dim>> solution_tensor(n_q_points);
                std::vector<SymmetricTensor<2, dim>> old_solution_tensor(n_q_points);

                fe_values[displacement].get_function_symmetric_gradients(solution, solution_tensor);
                fe_values[displacement].get_function_symmetric_gradients(old_solution, old_solution_tensor);

                // Retrieve the quadrature point history for this cell
                unsigned int cell_index = cell->active_cell_index() * n_q_points;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    std::shared_ptr<PointHistory<dim>> &qph = quadrature_point_history[cell_index + q_point];

                    qph->set_strain(solution_tensor[q_point]);

                    // retrieving D for step iii of Box 4.2 in the textbook
                    SymmetricTensor<4, dim> D_i = qph->get_consistent_tangent_operator();

                    // Compute the consistent tangent operator using the return mapping.
                    // The following is step vii (for the current step) and step ii (for the next step) in Box 4.2 in
                    // the textbook.
                    constitutive_law.return_mapping_and_derivative_stress_strain(solution_tensor[q_point] -
                                                                                old_solution_tensor[q_point], qph,
                                                                                yield_criteria);

                    SymmetricTensor<2, dim> stress_tensor = qph->get_stress();

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        if (!rhs_only)
                        {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                // the following is step iii in Box 4.2 in the textbook
                                cell_matrix(i, j) += transpose(fe_values[displacement].symmetric_gradient(i, q_point)) *
                                                     D_i * fe_values[displacement].symmetric_gradient(j, q_point) *
                                                     fe_values.JxW(q_point);
                            }
                        }

                        cell_rhs(i) -= transpose(fe_values[displacement].symmetric_gradient(i, q_point)) *
                                       stress_tensor * fe_values.JxW(q_point);
                    }
                }

                // Loop over faces of every cell
                for (const auto &face : cell->face_iterators())
                {
                    if (problem == "torsion")
                        if (face->at_boundary() && face->boundary_id() == 3)
                        {
                            fe_values_face.reinit(cell, face);
                            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                            {
                                EquationData::BoundaryForce<dim> boundary_force(problem, inner_radius, outer_radius,
                                                                                normal_force, torque);

                                Tensor<1, dim> traction;
                                traction = (1.0 / n_t_steps) * (t_step + 1) *
                                        boundary_force.value(fe_values_face.quadrature_point(q_point));

                                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                    cell_rhs(i) += fe_values_face[displacement].value(i, q_point) *
                                                   traction * fe_values_face.JxW(q_point);
                                }
                            }
                        }
//                    if (problem == "tensile")
//                        if (face->at_boundary() && face->boundary_id() == 5)
//                        {
//                            fe_values_face.reinit(cell, face);
//                            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
//                            {
//                                EquationData::BoundaryForce<dim> boundary_force(problem, inner_radius, outer_radius, normal_force, torque);
//
//                                Tensor<1, dim> traction;
//                                traction = (1.0 / n_t_steps) * (t_step + 1) * boundary_force.value(fe_values_face.quadrature_point(q_point));
//
//                                for (unsigned int i = 0; i < dofs_per_cell; ++i)
//                                {
//                                    cell_rhs(i) += fe_values_face[displacement].value(i, q_point) *
//                                                   traction * fe_values_face.JxW(q_point);
//                                }
//                            }
//                        }
//                    if (problem == "shear")
//                        if (face->at_boundary() && face->boundary_id() == 5)
//                        {
//                            fe_values_face.reinit(cell, face);
//                            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
//                            {
//                                EquationData::BoundaryForce<dim> boundary_force(problem, inner_radius, outer_radius, normal_force, torque);
//
//                                Tensor<1, dim> traction;
//                                traction = (1.0 / n_t_steps) * (t_step + 1) * boundary_force.value(fe_values_face.quadrature_point(q_point));
//
//                                for (unsigned int i = 0; i < dofs_per_cell; ++i)
//                                {
//                                    cell_rhs(i) += fe_values_face[displacement].value(i, q_point) *
//                                                   traction * fe_values_face.JxW(q_point);
//                                }
//                            }
//                        }
                }
                cell->get_dof_indices(local_dof_indices);
                all_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                                                           newton_matrix, newton_rhs);
            }
            newton_matrix.compress(VectorOperation::add);
            newton_rhs.compress(VectorOperation::add);
        }
    }


    template <int dim>
    void PlasticityProblem<dim>::solve_newton_system(TrilinosWrappers::MPI::Vector &newton_increment)
    {
        TimerOutput::Scope t(computing_timer, "Solve");

        // The following is a preconditioner setup. This accelerates the convergence of the iterative solver.
        TrilinosWrappers::PreconditionAMG preconditioner;
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
        {
            TimerOutput::Scope t(computing_timer, "Solve: iterate");

            TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

            const double relative_accuracy = 1e-8;
            const double solver_tolerance = relative_accuracy * newton_matrix.residual(tmp, newton_increment, newton_rhs);

            SolverControl solver_control(10 * newton_matrix.m(), solver_tolerance);

            SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

//            std::cout << "      Matrix norm      " << newton_matrix.frobenius_norm() << std::endl;
//            std::cout << "      RHS norm         " << newton_rhs.l2_norm() << std::endl;

            // Solve the system: newton_matrix * newton_increment = newton_rhs
            solver.solve(newton_matrix,
                         newton_increment,
                         newton_rhs,
                         preconditioner);

            // pcout << "         Error: " << solver_control.initial_value() << " -> "
            //       << solver_control.last_value() << " in "
            //       << solver_control.last_step() << " GMRES iterations."
            //       << std::endl;
        }
    }


    template <int dim>
    void PlasticityProblem<dim>::solve_newton(unsigned int n_time_steps, unsigned int t_step)
    {
        // TODO: create these once when setting up system for runtime efficiency
        // Declare newton_update vector (solution of a Newton iteration),
        // which must have as many positions as global DoFs.
        TrilinosWrappers::MPI::Vector newton_increment(locally_owned_dofs,
                                                           mpi_communicator);
        TrilinosWrappers::MPI::Vector tentative_solution(locally_owned_dofs,
                                                           mpi_communicator);
        // TODO: make Newton convergence criterion a parameter input
        const double tolerance = 1e-5;

        double previous_residual_norm;
        double current_residual_norm;

        // TODO: Make max number of steps an input. Throw error when does not converge
        unsigned int max_iterations_main_newton_loop = 250;
        for (unsigned int newton_step = 0; newton_step < max_iterations_main_newton_loop; ++newton_step)
        {
            pcout << ' ' << std::endl;
            pcout << "   Newton iteration " << newton_step << std::endl;

            bool main_newton_loop_converged = false;
            if (newton_step != 0)
            {
                std::cout << "      Norm at start of previous step: " << previous_residual_norm << std::endl;
            }

            // pcout << "      Assembling system... " << std::endl;
            newton_matrix = 0.;
            newton_rhs = 0.;

            if (newton_step != 0)
            {
                for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
                {
                    if (all_constraints.is_inhomogeneously_constrained(n))
                    {
                        all_constraints.set_inhomogeneity(n, 0);
                    }
                }
            }

            assemble_newton_system(solution, old_solution, false,n_time_steps, t_step);

            current_residual_norm = newton_rhs.l2_norm();
            std::cout << "      Norm based on solution from previous step: " << current_residual_norm << std::endl;

            newton_increment = 0;

            // pcout << "      Solving system... " << std::endl;
            solve_newton_system(newton_increment);

            all_constraints.distribute(newton_increment);

            // Explicitly check if this is an improvement
            double norm_tentative_soln;
            tentative_solution = solution;
            tentative_solution.add(1.0,newton_increment);
            bool line_search_successful = false;
            if (true || newton_step!=0) {
                // Line-search algorithm
                double alpha = 1.0;
                const double beta = 0.01; // Reduction factor for alpha

                while (alpha>= beta) {
                    tentative_solution = solution;
                    tentative_solution.add(1.0,newton_increment);
                    newton_rhs = 0;
                    assemble_newton_system(tentative_solution, old_solution, true,
                                                   n_time_steps, t_step); // Only assemble RHS (residual)

                    norm_tentative_soln = newton_rhs.l2_norm();
                    std::cout<<"      Norm of tentative new solution "<<norm_tentative_soln<<"with alpha of "<<alpha<<std::endl;
                    if (norm_tentative_soln<current_residual_norm || norm_tentative_soln<= tolerance) {
                        line_search_successful = true;
                        break;
                    }
                    alpha -= beta;
                }
            }
            if (line_search_successful)
                solution = tentative_solution;
            else
                AssertThrow(false, ExcMessage("Step does not make things better"));

            previous_residual_norm = current_residual_norm;

            if (std::abs(current_residual_norm) < tolerance)
                break;

            // AssertThrow(false, ExcMessage("Did not converge"));
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
    void PlasticityProblem<dim>::store_internal_variables()
    {
        const QGauss<dim> quadrature_formula(fe_degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points);

        const unsigned int n_q_points = quadrature_formula.size();

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);

                unsigned int cell_index = cell->active_cell_index() * n_q_points;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    quadrature_point_history[cell_index + q_point]->store_internal_variables();
                }
            }
        }
    }


    template <int dim>
    void PlasticityProblem<dim>::output_results(const unsigned int current_time_step,
                                                const std::string output_name)
    {
        TimerOutput::Scope t(computing_timer, "Graphical output");
        pcout << "      Writing graphical output... " << std::flush;

        move_mesh(solution);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        const QGauss<dim> quadrature_formula(fe.degree + 1);

        MappingQ<dim> mapping(fe_degree);

        const std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(solution, std::vector<std::string>(dim, "displacement"),
                                 DataOut<dim>::type_dof_data, data_component_interpretation);


        strain_tensor_tmp.reinit(locally_owned_scalar_dofs, mpi_communicator);
        for (int i = 0; i < dim; ++i)
            for (int j = i; j < dim; ++j)
            {
                VectorTools::project(mapping, dof_handler_scalar, scalar_constraints, quadrature_formula,
                                     [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         const unsigned int q_point) -> double
                                     {
                                         unsigned int local_index = cell->active_cell_index() *
                                                                    quadrature_formula.size();
                                         const std::shared_ptr<PointHistory<dim>> &lqph =
                                                 quadrature_point_history[local_index + q_point];
                                         const SymmetricTensor<2, dim> &e = lqph->get_strain();
                                         return e[i][j];
                                     },
                                     strain_tensor_tmp);

                std::string name = "e_" + std::to_string(i) + std::to_string(j);
                data_out.add_data_vector(dof_handler_scalar, strain_tensor_tmp, name);
            }


        accumulated_plastic_strain_vector.reinit(locally_owned_scalar_dofs, mpi_communicator);
        VectorTools::project(mapping, dof_handler_scalar, scalar_constraints, quadrature_formula,
                             [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 const unsigned int q_point) -> double
                             {
                                 unsigned int local_index = cell->active_cell_index() *
                                                            quadrature_formula.size();
                                 const std::shared_ptr<PointHistory<dim>> &lqph =
                                         quadrature_point_history[local_index + q_point];
                                 return lqph->get_accumulated_plastic_strain();
                             },
                             accumulated_plastic_strain_vector);
        data_out.add_data_vector(dof_handler_scalar, accumulated_plastic_strain_vector, "plastic_strain");


        is_plastic_vector.reinit(locally_owned_scalar_dofs, mpi_communicator);
        VectorTools::project(mapping, dof_handler_scalar, scalar_constraints, quadrature_formula,
                             [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 const unsigned int q_point) -> double
                             {
                                 unsigned int local_index = cell->active_cell_index() *
                                                            quadrature_formula.size();
                                 const std::shared_ptr<PointHistory<dim>> &lqph =
                                         quadrature_point_history[local_index + q_point];
                                 bool is_p = lqph->get_is_plastic();

                                 return is_p? 1.0 : 0.0;
                             },
                             is_plastic_vector);
        data_out.add_data_vector(dof_handler_scalar, is_plastic_vector, "plastic_indicator");


        stress_tensor_tmp.reinit(locally_owned_scalar_dofs, mpi_communicator);
        for (int i = 0; i < dim; ++i)
            for (int j = i; j < dim; ++j)
            {
                VectorTools::project(mapping, dof_handler_scalar, scalar_constraints, quadrature_formula,
                                     [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         const unsigned int q_point) -> double
                                     {
                                         unsigned int local_index = cell->active_cell_index() *
                                                                    quadrature_formula.size();
                                         const std::shared_ptr<PointHistory<dim>> &lqph =
                                                 quadrature_point_history[local_index + q_point];
                                         const SymmetricTensor<2, dim> &T = lqph->get_stress();
                                         return T[i][j];
                                     },
                                     stress_tensor_tmp);

                std::string name = "T_" + std::to_string(i) + std::to_string(j);
                data_out.add_data_vector(dof_handler_scalar, stress_tensor_tmp, name);
            }


        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        data_out.set_flags(flags);

        data_out.build_patches(mapping, fe_degree, DataOut<dim>::curved_inner_cells);  // this accomodates curved elements

        const std::string pvtu_filename =
                data_out.write_vtu_with_pvtu_record(output_dir,
                                                    output_name, current_time_step, mpi_communicator,
                                                    2);
        pcout << pvtu_filename << std::flush;

        TrilinosWrappers::MPI::Vector tmp(solution);
        tmp *= -1.0;
        move_mesh(tmp);

        std::cout << std::endl;
    }


    // The following function orchestrates the entire simulation. It manages mesh refinement, system solving
    // and output generation.
    template <int dim>
    void PlasticityProblem<dim>::run()
    {
        computing_timer.reset();

        make_grid(base_mesh);
        setup_system();

        constraints_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
        scalar_constraints.reinit(locally_owned_scalar_dofs, locally_relevant_scalar_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints_hanging_nodes);

        all_constraints.copy_from(constraints_dirichlet_and_hanging_nodes);
        all_constraints.close();
        scalar_constraints.close();

        constraints_hanging_nodes.close();
        constraints_dirichlet_and_hanging_nodes.close();

        output_results(0, "initial");

        output_results(0);

        double displacement = 0;
        double delta_displacement = 0;

        bool reverse_loading = false;

        unsigned int n_t_steps = n_time_steps;

        double delta_t;

        if (loading == "reverse")
        {
            delta_t = 3.0 / n_t_steps;
        }
        else
        {
            delta_t = 1.0 / n_t_steps;
        }

        for (unsigned int t_step = 0; t_step < n_t_steps; ++t_step)
        {
            std::cout << std::endl;

            std::cout << "Step: " << t_step << std::endl;

            if (reverse_loading == false && displacement >= applied_displacement)
            {
                reverse_loading = true;
            }

            if (reverse_loading == false)
            {
                delta_displacement = delta_t * applied_displacement;
            }
            else
            {
                delta_displacement = -delta_t * applied_displacement;
            }

            displacement += delta_displacement;

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

                compute_dirichlet_constraints(delta_displacement, problem);

                all_constraints.copy_from(constraints_dirichlet_and_hanging_nodes);
                all_constraints.close();
                scalar_constraints.close();

                constraints_hanging_nodes.close();
                constraints_dirichlet_and_hanging_nodes.close();
            }

            solve_newton(n_t_steps, t_step);

            store_internal_variables();

            old_solution = solution;

            output_results(t_step + 1);
        }

        computing_timer.print_summary();
        computing_timer.reset();

        Utilities::System::MemoryStats stats;
        Utilities::System::get_memory_stats(stats);
        pcout << "Peak virtual memory used, resident in kB: " << stats.VmSize
              << ' ' << stats.VmRSS << std::endl;
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
