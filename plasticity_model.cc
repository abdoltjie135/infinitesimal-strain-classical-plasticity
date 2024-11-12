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


// TODO: Place the utility functions in a separate file


namespace PlasticityModel
{
    using namespace dealii;

    // TODO: This needs to be setup to work with multiple cycles
    // FIXME: The order in which I declare variables/functions and define functions might not be the best but it
    //  is fine for now
    template <int dim>
    class PointHistory
    {
    public:
        PointHistory();

        void store_internal_variables();

        // A setter allows controlled modification of a private or protected member variable, typically
        // ensuring that the data being set is valid or consistent
        void set_stress(const SymmetricTensor<2, dim> &stress);
        void set_elastic_strain(const SymmetricTensor<2, dim> &elastic_strain);
        void set_plastic_strain(const SymmetricTensor<2, dim> &plastic_strain);
        void set_back_stress(const SymmetricTensor<2, dim> &back_stress);
        void set_consistent_tangent_operator(const SymmetricTensor<4, dim> &consistent_tangent_operator);
        void set_principal_stresses(const std::vector<double> &principal_stresses);
        void set_accumulated_plastic_strain(double accumulated_plastic_strain);
        void set_is_plastic(bool is_plastic);

        // A getter retrieves (or "gets") the value of a private or protected member variable from the class
        // without allowing external code to modify it directly
        SymmetricTensor<2, dim> get_stress() const;
        SymmetricTensor<2, dim> get_elastic_strain() const;
        SymmetricTensor<2, dim> get_plastic_strain() const;
        SymmetricTensor<2, dim> get_back_stress() const;
        SymmetricTensor<4, dim> get_consistent_tangent_operator() const;
        std::vector<double> get_principal_stresses() const;
        // NOTE: I am not sure if the following two functions are needed
        double get_accumulated_plastic_strain() const;
        bool get_is_plastic();

    private:
        // TODO: Create getters and setters for these variables
        SymmetricTensor<2, dim> stress;
        SymmetricTensor<2, dim> elastic_strain;
        SymmetricTensor<2, dim> plastic_strain;
        SymmetricTensor<2, dim> back_stress;
        SymmetricTensor<4, dim> consistent_tangent_operator;
        std::vector<double> principal_stresses;
        double accumulated_plastic_strain;
        bool is_plastic;

        SymmetricTensor<2, dim> stored_stress;
        SymmetricTensor<2, dim> stored_elastic_strain;
        SymmetricTensor<2, dim> stored_plastic_strain;
        SymmetricTensor<2, dim> stored_back_stress;
        SymmetricTensor<4, dim> stored_consistent_tangent_operator;
        std::vector<double> stored_principal_stresses;
    };

    // Constructor definition
    template <int dim>
    PointHistory<dim>::PointHistory()
        : consistent_tangent_operator(SymmetricTensor<4, dim>()) // Initialize with default value
    {
        const double shear_modulus = 76920.0;
        const double bulk_modulus = 166670.0;

        // setting the initial value of the consistent tangent operator to the elastic consistent tangent operator
        consistent_tangent_operator = 2.0 * shear_modulus *
            (identity_tensor<dim>() - 1.0 / 3.0 *
                outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())) +
                    bulk_modulus * outer_product(unit_symmetric_tensor<dim>(),
                        unit_symmetric_tensor<dim>());

        is_plastic = false;
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
    void PointHistory<dim>::set_plastic_strain(const SymmetricTensor<2, dim> &plastic_strain)
    {
        this->plastic_strain = plastic_strain;
    }

    template <int dim>
    void PointHistory<dim>::set_back_stress(const SymmetricTensor<2, dim> &back_stress)
    {
        this->back_stress = back_stress;
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
    void PointHistory<dim>::set_accumulated_plastic_strain(double accumulated_plastic_strain)
    {
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
    SymmetricTensor<2, dim> PointHistory<dim>::get_elastic_strain() const
    {
        return elastic_strain;
    }

    template<int dim>
    SymmetricTensor<2, dim> PointHistory<dim>::get_plastic_strain() const
    {
        return plastic_strain;
    }

    template<int dim>
    SymmetricTensor<2, dim> PointHistory<dim>::get_back_stress() const
    {
        return back_stress;
    }

    template<int dim>
    std::vector<double> PointHistory<dim>::get_principal_stresses() const
    {
        return principal_stresses;
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
        stored_back_stress = back_stress;
        stored_consistent_tangent_operator = consistent_tangent_operator;
        stored_principal_stresses = principal_stresses;
    }


    template <int dim>
    class ConstitutiveLaw
    {
    public:
        ConstitutiveLaw(const double sigma_0,                   // initial yield stress
                        const double hardening_slope,           // hardening slope (H in the textbook)
                        const double kappa,                     // bulk modulus (K in the textbook)
                        const double mu);                       // shear modulus (G in the textbook)

        // The following function performs the return-mapping algorithm and determines the derivative of stress with
        // respect to strain based off of whether a one-vector or a two-vector return was used
        bool return_mapping_and_derivative_stress_strain(const SymmetricTensor<2, dim>& elastic_strain_trial,
            std::shared_ptr<PointHistory<dim>> &qph,
            std::string yield_criteria) const;

        SymmetricTensor<4, dim> derivative_of_isotropic_tensor(Tensor<1, dim> x, Tensor<2, dim> e,
            SymmetricTensor<2, dim>, SymmetricTensor<2, dim> Y, SymmetricTensor<2, dim> dy_dx) const;

    private:
        const double sigma_0;
        const double kappa;
        const double mu;
        const double hardening_slope;

        // SymmetricTensor<4, dim> elastic_consistent_tangent_operator;
    };


    // The following is the definition of the constructor
    template <int dim>
    ConstitutiveLaw<dim>::ConstitutiveLaw(const double sigma_0,
                                          const double hardening_slope,
                                          const double kappa,
                                          const double mu)
    // initialize the member variables
    : sigma_0(sigma_0)
    , kappa(kappa)
    , mu(mu)
    , hardening_slope(hardening_slope)
    {} // constructor body is empty because all the work is done in the initializer list


    // Function to compute and sort eigenvalues and eigenvectors of a tensor in descending order
    template <int dim>
    std::pair<Tensor<1, dim>, Tensor<2, dim>> compute_principal_values_vectors(const SymmetricTensor<2, dim> &A)
    {
        // Find eigenvalues and eigenvectors
        auto eigenvector_pairs = eigenvectors(A);

        // Sort eigenvalues and eigenvectors in descending order
        std::sort(eigenvector_pairs.begin(), eigenvector_pairs.end(),
                  [](const std::pair<double, Tensor<1, dim>> &a, const std::pair<double, Tensor<1, dim>> &b) {
                      return a.first > b.first;
                  });

        // Extract sorted eigenvalues and eigenvectors
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


    // The following function checks if it is plastic or elastic
    template <int dim>
    bool ConstitutiveLaw<dim>::return_mapping_and_derivative_stress_strain(
        const SymmetricTensor<2, dim> &elastic_strain_trial,
        std::shared_ptr<PointHistory<dim>> &qph,
        std::string yield_criteria) const
    {
        Assert(dim == 3, ExcNotImplemented());

        // Material properties for consistent tangent computation
        const double yield_stress_0 = sigma_0;
        const double G = mu;
        const double K = kappa;
        const double H = hardening_slope;

        double accumulated_plastic_strain_n;

        // Elastic Predictor Step (Box 8.1, Step i) from the textbook
        auto elastic_strain_trial_eigenvector_pairs =
            compute_principal_values_vectors(elastic_strain_trial);
        auto elastic_strain_trial_eigenvalues = elastic_strain_trial_eigenvector_pairs.first;
        auto elastic_strain_trial_eigenvectors_matrix = elastic_strain_trial_eigenvector_pairs.second;

        // NOTE: Debugging output
        // std::cout << "   Eigenvector Matrix of the elastic strain trial: " << elastic_strain_trial_eigenvectors_matrix << std::endl;
        // std::cout << "   Eigenvalues of the elastic strain trial: " << elastic_strain_trial_eigenvalues << std::endl;

        double accumulated_plastic_strain_trial = accumulated_plastic_strain_n;  // ε_p_n+1^trial
        SymmetricTensor<2, dim> deviatoric_stress_trial = 2.0 * G *              // deviatoric stress trial
            deviator(elastic_strain_trial);
        double e_v_trial = trace(elastic_strain_trial);                          // volumetric part of the trial elastic strain
        double p_trial = K * e_v_trial;

        SymmetricTensor<2, dim> ds_de;
        SymmetricTensor<2, dim> dsigma_de;

        // Initializing state variables at n+1
        SymmetricTensor<2, dim> elastic_strain_n1;
        SymmetricTensor<2, dim> plastic_strain_n1;     // This is not used in the current implementation
        SymmetricTensor<2, dim> deviatoric_stress_n1;
        double accumulated_plastic_strain_n1;
        SymmetricTensor<2, dim> s_n1;                  // deviatoric stress at n+1 in the principal directions
        SymmetricTensor<2, dim> stress_n1;             // p_n+1^trial

        // Spectral decomposition (Box 8.1, Step ii) from the textbook
        auto deviatoric_stress_trial_eigenvalues = compute_principal_values(deviatoric_stress_trial);

        double s1, s2, s3;  // declare principal deviatoric stresses

        double tolerance = 1e-6;  // tolerance for Newton iterations

        double delta_gamma;

        double q_trial;

        SymmetricTensor<4, dim> consistent_tangent_operator;

        if (yield_criteria == "Tresca")
        {
            // Plastic admissibility check (Box 8.1, Step iii) from the textbook
            double phi = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[dim - 1] -
                yield_stress(yield_stress_0, H, accumulated_plastic_strain_trial);

            if (phi <= 0)
            {
                // std::cout << "Elastic step" << std::endl;

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

                // NOTE: The stress is in the principal directions therefore you might have to reconstruct it back to the
                //  original co-ordinate system before storing and outputting it here.
                //  You can easily store the principal values below because they are already in the principal directions
                //  therefore be careful where you make the reconstruction.
                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                SymmetricTensor<2, dim> stress_n1_reconstructed = reconstruct_tensor({{stress_n1[0][0],
                    stress_n1[1][1], stress_n1[2][2]}}, elastic_strain_trial_eigenvectors_matrix);

                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                // no change to the consistent tangent operator for elastic step

                return false;
            }

            qph->set_is_plastic(true);

            // Plastic step: Return Mapping (Box 8.1, Step iv) from the textbook
            // double delta_gamma;
            double residual = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[dim - 1] -
               yield_stress(yield_stress_0, H, accumulated_plastic_strain_n);

            // Initially say it is going to be a two-vector return
            bool local_newton_converged_one_v = false;

            // One-vector return to main plane (Box 8.2) from the textbook
            for (unsigned int iteration = 0; iteration < 50; ++iteration)
            {
                // Update yield stress after accumulating plastic strain (Box 8.2, Step ii)
                // TODO: For general isotropic hardening the hardening slope (hardening_modulus) needs to be determined
                //  for linear isotropic hardening it is constant
                //  see box 8.2 in Computational Methods for Plasticity on how to determine the hardening slope
                double residual_derivative = -4.0 * G - H;
                delta_gamma -= residual / residual_derivative;  // new guess for delta_gamma

                // Compute the residual (Box 8.2, Step ii)
                residual = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[dim - 1] -
                    4.0 * G * delta_gamma -
                        yield_stress(yield_stress_0, H, accumulated_plastic_strain_n + delta_gamma);;

                if (std::abs(residual) <= tolerance)
                {
                    // Update principal deviatoric stresses (Box 8.2, Step iii)
                    s1 = deviatoric_stress_trial_eigenvalues[0] - 2.0 * G * delta_gamma;
                    s2 = deviatoric_stress_trial_eigenvalues[1];
                    s3 = deviatoric_stress_trial_eigenvalues[2] + 2.0 * G * delta_gamma;

                    accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma;

                    local_newton_converged_one_v = true;

                    break;
                }
            }

            // Checking if the Newton iteration converged
            AssertThrow(local_newton_converged_one_v,
                ExcMessage("Newton iteration did not converge for one-vector return"));

            /*
            // NOTE: Solving in closed form because the one-vector return is not converging
            delta_gamma = (s_trial_eigenvalues[0] - s_trial_eigenvalues[dim - 1] - yield_stress_0 - H *
                accumulated_plastic_strain_n) / (4 * G + H);

            // Update principal deviatoric stresses (Box 8.2, Step iii)
            s1 = s_trial_eigenvalues[0] - 2.0 * G * delta_gamma;
            s2 = s_trial_eigenvalues[1];
            s3 = s_trial_eigenvalues[2] + 2.0 * G * delta_gamma;

            accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma;
            */

            // Check if the updated principal stresses satisfy s1 >= s2 >= s3 (Box 8.1, Step iv.b) from the textbook
            if (s1 >= s2 && s2 >= s3)
            {
                // one-vector return

                // std::cout << "Plastic: one-vector return" << std::endl;

                double f = (2.0 * G) / (4.0 * G + H);

                // This relates to equation 8.4.1 in the textbook
                ds_de[0][0] = 2.0 * G * (1.0 - f);
                ds_de[0][2] = 2.0 * G * f;
                ds_de[1][1] = 2.0 * G;
                ds_de[2][0] = 2.0 * G * f;
                ds_de[2][2] = 2.0 * G * (1.0 - f);

                // Equation 8.46 from textbook
                for (unsigned int i = 0; i < 3; ++i) // Iterate over i (curly brackets after this for loop are not needed)
                    for (unsigned int j = 0; j < 3; ++j) // Iterate over j
                    {
                        dsigma_de[i][j] = K;

                        // Loop through the k index for the sum over k
                        for (unsigned int k = 0; k < 3; ++k)
                        {
                            double delta_kj = (k == j) ? 1.0 : 0.0;
                            dsigma_de[i][j] += ds_de[i][k] * (delta_kj - 1.0 / 3.0);
                        }
                    }

                qph->set_principal_stresses({s1, s2, s3});

                double p_n1 = p_trial;

                s_n1[0][0] = s1;
                s_n1[1][1] = s2;
                s_n1[2][2] = s3;

                // NOTE: The stress is in the principal directions therefore you might have to reconstruct it back to the
                //  original co-ordinate system before storing and outputting it here.
                //  You can easily store the principal values below because they are already in the principal directions
                //  therefore be careful where you make the reconstruction.
                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                SymmetricTensor<2, dim> stress_n1_reconstructed = reconstruct_tensor({{stress_n1[0][0],
                    stress_n1[1][1], stress_n1[2][2]}}, elastic_strain_trial_eigenvectors_matrix);

                // NOTE: The elastic strain has be reconstructed back to the original co-ordinate system before storing
                //  I assumed that th eigenvectors of the eigenvectors of the deviatoric part of the stress is the same
                //  as the full stress tensor
                elastic_strain_n1 = (1.0 / 2.0 * G) *
                    reconstruct_tensor({{s1, s2, s3}}, elastic_strain_trial_eigenvectors_matrix) +
                    (1.0 / 3.0) * e_v_trial * unit_symmetric_tensor<dim>();

                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                consistent_tangent_operator = derivative_of_isotropic_tensor(elastic_strain_trial_eigenvalues,
                    elastic_strain_trial_eigenvectors_matrix, elastic_strain_trial, stress_n1, dsigma_de);

                qph->set_consistent_tangent_operator(consistent_tangent_operator);

                return true;
            }

            // Two-vector return

            // std::cout << "Plastic: two-vector return" << std::endl;

            if (deviatoric_stress_trial_eigenvalues[0] + deviatoric_stress_trial_eigenvalues[dim - 1] -
                2.0 * deviatoric_stress_trial_eigenvalues[1] > 0.)
            {
                // Right corner return

                // computing ds/de for the two-vector right return
                // This relates to equation 8.5.2 in the textbook
                double daa = -4.0 * G - H;
                double dab = -2.0 * G - H;
                double dba = -2.0 * G - H;
                double dbb = -4.0 * G - H;

                double det_d = daa * dbb - dab * dba;

                ds_de[0][0] = 2.0 * G * (1.0 - (8.0 * pow(G, 2.0)) / det_d);
                ds_de[0][1] = ((4.0 * pow(G, 2.0)) / det_d) * (dab - daa);
                ds_de[0][2] = (4.0 * pow(G, 2.0) / det_d) * (dba - dbb);
                ds_de[1][0] = (8.0 * pow(G, 3.0)) / det_d;
                ds_de[1][1] = 2.0 * G * (1.0 + (2.0 * G * daa) / det_d);
                ds_de[1][2] = -(4.0 * pow(G, 2.0) / det_d) * dba;
                ds_de[2][0] = (8.0 * pow(G, 3.0)) / det_d;
                ds_de[2][1] = -(4.0 * pow(G, 2.0) / det_d) * dab;
                ds_de[2][2] = 2.0 * G * (1.0 + (2.0 * G * dbb) / det_d);

                // Equation 8.46 from textbook
                for (unsigned int i = 0; i < 3; ++i) // Iterate over i (curly brackets after this for loop are not needed)
                    for (unsigned int j = 0; j < 3; ++j) // Iterate over j
                    {
                        dsigma_de[i][j] = K;

                        // Loop through the k index for the sum over k
                        for (unsigned int k = 0; k < 3; ++k)
                        {
                            double delta_kj = (k == j) ? 1.0 : 0.0;
                            dsigma_de[i][j] += ds_de[i][k] * (delta_kj - 1.0 / 3.0);
                        }
                    }

                Vector<double> delta_gamma_vector(2);
                delta_gamma_vector[0] = 0.0;
                delta_gamma_vector[1] = 0.0;

                double s_b = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[1];
                double s_a = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[2];

                // NOTE: The function for the yield stress will look different for nonlinear isotropic hardening
                Vector<double> residual_vector(2);
                residual_vector[0] = s_a - yield_stress(yield_stress_0, H,
                    accumulated_plastic_strain_n);
                residual_vector[1] = s_b - yield_stress(yield_stress_0, H,
                    accumulated_plastic_strain_n);

                Vector<double> delta_gamma_vector_update(2);

                // Newton iteration for two-vector return (Box 8.3, Step ii) from the textbook
                for (unsigned int iteration = 0; iteration < 50; ++iteration)
                {
                    double delta_gamma_sum = delta_gamma_vector[0] + delta_gamma_vector[1];
                    double accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma_sum;

                    // TODO: The hardening slope is constant for linear isotropic hardening
                    //  this will need to be made general for general isotropic hardening later

                    FullMatrix<double> d_matrix(2, 2);
                    d_matrix(0, 0) = -4.0 * G - H;
                    d_matrix(0, 1) = -2.0 * G - H;
                    d_matrix(1, 0) = -2.0 * G - H;
                    d_matrix(1, 1) = -4.0 * G - H;

                    FullMatrix<double> d_matrix_inverse(2, 2);
                    d_matrix_inverse.invert(d_matrix);

                    residual_vector *= -1.0;
                    d_matrix_inverse.vmult(delta_gamma_vector_update, residual_vector);

                    delta_gamma_vector += delta_gamma_vector_update;

                    residual_vector[0] = s_a - 2.0 * G * (2.0 * delta_gamma_vector[0] + delta_gamma_vector[1]) -
                        yield_stress(sigma_0, H, accumulated_plastic_strain_n1);
                    residual_vector[1] = s_b - 2.0 * G * (delta_gamma_vector[0] + 2 * delta_gamma_vector[1]) -
                        yield_stress(sigma_0, H, accumulated_plastic_strain_n1);

                    if (abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance)
                    {
                        s1 = deviatoric_stress_trial_eigenvalues[0] - 2.0 * G * (delta_gamma_vector[0] +
                            delta_gamma_vector[1]);
                        s2 = deviatoric_stress_trial_eigenvalues[1] + 2.0 * G * delta_gamma_vector[1];
                        s3 = deviatoric_stress_trial_eigenvalues[2] + 2.0 * G * delta_gamma_vector[0];

                        break;
                    }
                }

                AssertThrow(abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance,
                    ExcMessage("Two-vector return did not converge"));

                qph->set_principal_stresses({s1, s2, s3});

                double p_n1 = p_trial;

                s_n1[0][0] = s1;
                s_n1[1][1] = s2;
                s_n1[2][2] = s3;

                // NOTE: The stress is in the principal directions therefore you might have to reconstruct it back to the
                //  original co-ordinate system before storing and outputting it here.
                //  You can easily store the principal values below because they are already in the principal directions
                //  therefore be careful where you make the reconstruction.
                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                SymmetricTensor<2, dim> stress_n1_reconstructed = reconstruct_tensor({{stress_n1[0][0],
                    stress_n1[1][1], stress_n1[2][2]}}, elastic_strain_trial_eigenvectors_matrix);

                // NOTE: The elastic strain has be reconstructed back to the original co-ordinate system before storing
                //  I assumed that th eigenvectors of the eigenvectors of the deviatoric part of the stress is the same
                //  as the full stress tensor
                elastic_strain_n1 = (1.0 / 2.0 * G) *
                    reconstruct_tensor({{s1, s2, s3}}, elastic_strain_trial_eigenvectors_matrix) +
                    (1.0 / 3.0) * e_v_trial * unit_symmetric_tensor<dim>();

                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                consistent_tangent_operator = derivative_of_isotropic_tensor(elastic_strain_trial_eigenvalues,
                    elastic_strain_trial_eigenvectors_matrix, elastic_strain_trial, stress_n1, dsigma_de);

                qph->set_consistent_tangent_operator(consistent_tangent_operator);

                return true;
            }

            // Left corner return

            // computing ds/de for the two-vector right return
            // This relates to equation 8.5.3 in the textbook
            double daa = -4.0 * G - H;
            double dab = -2.0 * G - H;
            double dba = -2.0 * G - H;
            double dbb = -4.0 * G - H;

            double det_d = daa * dbb - dab * dba;

            ds_de[0][0] = 2.0 * G * (1.0 + (2.0 * G * dbb) / det_d);
            ds_de[0][1] = -(4.0 * pow(G, 2.0) / det_d) * dab;
            ds_de[0][2] = (8.0 * pow(G, 3.0)) / det_d;
            ds_de[1][0] = -(4.0 * pow(G, 2.0) / det_d) * dba;
            ds_de[1][1] = 2.0 * G * (1.0 + (2.0 * G * daa) / det_d);
            ds_de[1][2] = (8.0 * pow(G, 3.0)) / det_d;
            ds_de[2][0] = (4.0 * pow(G, 2.0) / det_d) * (dba - dbb);
            ds_de[2][1] = (4.0 * pow(G, 2.0) / det_d) * (dab - daa);
            ds_de[2][2] = 2.0 * G * (1.0 - (8.0 * pow(G, 2.0)) / det_d);

            // Equation 8.46 from textbook
            for (unsigned int i = 0; i < 3; ++i) // Iterate over i (curly brackets after this for loop are not needed)
                for (unsigned int j = 0; j < 3; ++j) // Iterate over j
                {
                    dsigma_de[i][j] = K;

                    // Loop through the k index for the sum over k
                    for (unsigned int k = 0; k < 3; ++k)
                    {
                        double delta_kj = (k == j) ? 1.0 : 0.0;
                        dsigma_de[i][j] += ds_de[i][k] * (delta_kj - 1.0 / 3.0);
                    }
                }

            Vector<double> delta_gamma_vector(2);
            delta_gamma_vector[0] = 0.0;
            delta_gamma_vector[1] = 0.0;

            double s_b = deviatoric_stress_trial_eigenvalues[1] - deviatoric_stress_trial_eigenvalues[2];

            double s_a = deviatoric_stress_trial_eigenvalues[0] - deviatoric_stress_trial_eigenvalues[2];

            // NOTE: The function for the yield stress will look different for nonlinear isotropic hardening
            Vector<double> residual_vector(2);
            residual_vector[0] = s_a - yield_stress(yield_stress_0, H,
                accumulated_plastic_strain_n);
            residual_vector[1] = s_b - yield_stress(yield_stress_0, H,
                accumulated_plastic_strain_n);

            Vector<double> delta_gamma_vector_update(2);

            // Newton iteration for two-vector return (Box 8.3, Step ii) from the textbook
            for (unsigned int iteration = 0; iteration < 50; ++iteration)
            {
                double delta_gamma_sum = delta_gamma_vector[0] + delta_gamma_vector[1];
                double accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma_sum;

                // TODO: The hardening slope is constant for linear isotropic hardening
                //  this will need to be made general for general isotropic hardening later

                FullMatrix<double> d_matrix(2, 2);
                d_matrix(0, 0) = -4.0 * G - H;
                d_matrix(0, 1) = -2.0 * G - H;
                d_matrix(1, 0) = -2.0 * G - H;
                d_matrix(1, 1) = -4.0 * G - H;

                FullMatrix<double> d_matrix_inverse(2, 2);
                d_matrix_inverse.invert(d_matrix);

                residual_vector *= -1.0;
                d_matrix_inverse.vmult(delta_gamma_vector_update, residual_vector);

                delta_gamma_vector += delta_gamma_vector_update;

                residual_vector[0] = s_a - 2.0 * G * (2.0 * delta_gamma_vector[0] + delta_gamma_vector[1]) -
                    (yield_stress_0 + H * accumulated_plastic_strain_n1);
                residual_vector[1] = s_b - 2.0 * G * (2.0 * delta_gamma_vector[1] + delta_gamma_vector[0]) -
                    (yield_stress_0 + H * accumulated_plastic_strain_n1);

                if (abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance)
                {
                    s1 = deviatoric_stress_trial_eigenvalues[0] - 2.0 * G * delta_gamma_vector[0];
                    s2 = deviatoric_stress_trial_eigenvalues[1] - 2.0 * G * delta_gamma_vector[1];
                    s3 = deviatoric_stress_trial_eigenvalues[2] + 2.0 * G * (delta_gamma_vector[0] +
                        delta_gamma_vector[1]);

                    break;
                }
            }

            AssertThrow(abs(residual_vector[0]) + abs(residual_vector[1]) <= tolerance,
                ExcMessage("Two-vector return did not converge"));

            qph->set_principal_stresses({s1, s2, s3});

            double p_n1 = p_trial;

            s_n1[0][0] = s1;
            s_n1[1][1] = s2;
            s_n1[2][2] = s3;

            // NOTE: The stress is in the principal directions therefore you might have to reconstruct it back to the
            //  original co-ordinate system before storing and outputting it here.
            //  You can easily store the principal values below because they are already in the principal directions
            //  therefore be careful where you make the reconstruction.
            stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

            SymmetricTensor<2, dim> stress_n1_reconstructed = reconstruct_tensor({{stress_n1[0][0],
                stress_n1[1][1], stress_n1[2][2]}}, elastic_strain_trial_eigenvectors_matrix);

            // NOTE: The elastic strain has be reconstructed back to the original co-ordinate system before storing
            //  I assumed that th eigenvectors of the eigenvectors of the deviatoric part of the stress is the same
            //  as the full stress tensor
            elastic_strain_n1 = (1.0 / 2.0 * G) *
                reconstruct_tensor({{s1, s2, s3}}, elastic_strain_trial_eigenvectors_matrix) +
                (1.0 / 3.0) * e_v_trial * unit_symmetric_tensor<dim>();

            qph->set_stress(stress_n1_reconstructed);
            qph->set_elastic_strain(elastic_strain_n1);
            qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

            consistent_tangent_operator = derivative_of_isotropic_tensor(elastic_strain_trial_eigenvalues,
                    elastic_strain_trial_eigenvectors_matrix, elastic_strain_trial, stress_n1, dsigma_de);

            qph->set_consistent_tangent_operator(consistent_tangent_operator);

            return true;
        }
        if (yield_criteria == "Von-Mises")
        {
            q_trial = sqrt((3.0 / 2.0) * scalar_product(deviatoric_stress_trial, deviatoric_stress_trial)); // q_n+1^trial

            if (q_trial - yield_stress(yield_stress_0, H, accumulated_plastic_strain_trial) <= 0.0)
            {
                // std::cout << "Elastic step" << std::endl;

                // TODO: Set values at n+1 to n+1^trial
                // Elastic step therefore setting values at n+1 to trial values
                elastic_strain_n1 = elastic_strain_trial;
                accumulated_plastic_strain_n1 = accumulated_plastic_strain_trial;

                double p_n1 = p_trial;

                // deviatoric stress in the principal directions
                s1 = deviatoric_stress_trial_eigenvalues[0];
                s2 = deviatoric_stress_trial_eigenvalues[1];
                s3 = deviatoric_stress_trial_eigenvalues[2];

                qph->set_principal_stresses({s1, s2, s3});

                s_n1[0][0] = s1;
                s_n1[1][1] = s2;
                s_n1[2][2] = s3;

                stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

                SymmetricTensor<2, dim> stress_n1_reconstructed = reconstruct_tensor({{stress_n1[0][0],
                stress_n1[1][1], stress_n1[2][2]}}, elastic_strain_trial_eigenvectors_matrix);

                // elastic
                consistent_tangent_operator = 2.0 * G * (identity_tensor<dim>() -
                    1.0 / 3.0 * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())) +
                        K * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());

                qph->set_consistent_tangent_operator(consistent_tangent_operator);
                qph->set_stress(stress_n1_reconstructed);
                qph->set_elastic_strain(elastic_strain_n1);
                qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

                return false;  // it is elastic
            }

            qph->set_is_plastic(true);

            // std::cout << "Plastic step" << std::endl;

            double residual = q_trial - yield_stress(yield_stress_0, H, accumulated_plastic_strain_n);

            for (unsigned int iteration = 0; iteration < 50; ++iteration)
            {
                double residual_derivative = -3.0 * G - H;
                delta_gamma -= residual / residual_derivative;  // new guess for delta_gamma

                residual = q_trial - 3.0 * G * delta_gamma -
                    yield_stress(yield_stress_0, H, accumulated_plastic_strain_n + delta_gamma);

                if (std::abs(residual) < tolerance)
                {
                    break;
                }
            }

            AssertThrow(std::abs(residual) < tolerance,
            ExcMessage("Newton iteration did not converge for Von-Mises yield criteria"));

            // Elastic step therefore setting values at n+1 to trial values
            accumulated_plastic_strain_n1 = accumulated_plastic_strain_n + delta_gamma;

            // deviatoric stress in the principal directions
            s_n1[0][0] = (1.0 - (delta_gamma * 3.0 * G) / q_trial) * deviatoric_stress_trial_eigenvalues[0];
            s_n1[1][1] = (1.0 - (delta_gamma * 3.0 * G) / q_trial) * deviatoric_stress_trial_eigenvalues[1];
            s_n1[2][2] = (1.0 - (delta_gamma * 3.0 * G) / q_trial) * deviatoric_stress_trial_eigenvalues[2];

            qph->set_principal_stresses({s_n1[0][0], s_n1[1][1], s_n1[2][2]});

            double p_n1 = p_trial;

            stress_n1 = s_n1 + p_n1 * unit_symmetric_tensor<dim>();

            SymmetricTensor<2, dim> stress_n1_reconstructed = reconstruct_tensor({{stress_n1[0][0],
                stress_n1[1][1], stress_n1[2][2]}}, elastic_strain_trial_eigenvectors_matrix);

            elastic_strain_n1 = (1.0 / 2.0 * G) *
                reconstruct_tensor({{s_n1[0][0], s_n1[1][1], s_n1[2][2]}}, elastic_strain_trial_eigenvectors_matrix) +
                (1.0 / 3.0) * e_v_trial * unit_symmetric_tensor<dim>();

            qph->set_stress(stress_n1_reconstructed);
            qph->set_elastic_strain(elastic_strain_n1);
            qph->set_accumulated_plastic_strain(accumulated_plastic_strain_n1);

            SymmetricTensor<2, dim> N_n1 = deviatoric_stress_trial / deviatoric_stress_trial.norm();

            consistent_tangent_operator = 2.0 * G * (1.0 - (delta_gamma * 3.0 * G) / q_trial) * (identity_tensor<dim>() -
                1.0 / 3.0 * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())) +
                    6.0 * G * G * (delta_gamma / q_trial - 1.0 / (3.0 * G + H)) * outer_product(N_n1, N_n1) +
                        K * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());

            // NOTE: I am not sure if the consistent tangent operator needs to be transformed back to th original co-ordinate
            //  system
            qph->set_consistent_tangent_operator(consistent_tangent_operator);

            return true;
        }
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

        // Calculate the 4th-order tensor d[X^2]/dX_ijkl as per equation A.46 in the textbook
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
        std::array<SymmetricTensor<2, dim>, 3> E;

        // Initialize the 4th-order tensor D to zero
        SymmetricTensor<4, dim> D;

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

            // TODO: These need to be checked if they are copied (from the textbook) correctly
            // The following are not deviatoric stress values
            double s1 = (y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c])) +
                        (1.0 / (x[a] - x[c])) * (dy_dx[c][b] - dy_dx[c][c]);

            double s2 = 2.0 * x[c] * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]))) +
                       ((x[a] + x[c]) / (x[a] - x[c])) * (dy_dx[c][b] - dy_dx[c][c]);

            double s3 = 2.0 * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c]))) +
                       (1.0 / ((x[a] - x[c]) * (x[a] - x[c]))) * (dy_dx[a][c] + dy_dx[c][a] - dy_dx[a][a] - dy_dx[c][c]);

            double s4 = 2.0 * x[c] * (y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c])) +
                       (1.0 / (x[a] - x[c])) *
                       (dy_dx[a][c] - dy_dx[c][b]) + (x[c] / ((x[a] - x[c]) * (x[a] - x[c]))) *
                       (dy_dx[a][c] + dy_dx[c][a] - dy_dx[a][a] - dy_dx[c][c]);

            double s5 = 2.0 * x[c] * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c]))) +
                       (1.0 / (x[a] - x[c])) * (dy_dx[c][a] - dy_dx[c][b]) +
                           (x[c] / ((x[a] - x[c]) * (x[a] - x[c]))) * (dy_dx[a][c] + dy_dx[c][a] - dy_dx[a][a] - dy_dx[c][c]);

            double s6 = 2.0 * (x[c] * x[c]) * ((y[a] - y[c]) / ((x[a] - x[c]) * (x[a] - x[c]) * (x[a] - x[c]))) +
                       ((x[a] * x[c]) / ((x[a] - x[c]) * (x[a] - x[c]))) *
                       (dy_dx[a][c] + dy_dx[c][a])  - ((x[c] * x[c]) / ((x[a] - x[c]) * (x[a] - x[c]))) *
                       (dy_dx[a][a] + dy_dx[c][c]) - ((x[a] + x[c]) / (x[a] - x[c])) * dy_dx[c][b];

            D = s1 * dX2_dX - s2 * identity_tensor<dim>() - s3 * outer_product(X, X) +
                s4 * outer_product(X, unit_symmetric_tensor<dim>()) +
                    s5 * outer_product(unit_symmetric_tensor<dim>(), X) -
                    s6 * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
        }
        else if (x[0] == x[1] && x[1] == x[2])
        {
            D = (dy_dx[0][0] - dy_dx[0][1]) * identity_tensor<dim>() +
                dy_dx[0][1] * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>());
        }

        return D;
    }


    namespace EquationData
    {
        template <int dim>
        class BoundaryForce : public Function<dim>
        {
        public:
            BoundaryForce();

            virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;

            virtual void vector_value(const Point<dim>& p, Vector<double>& values) const override;
        };

        template <int dim>
        BoundaryForce<dim>::BoundaryForce()  // constructor definition
            : Function<dim>(dim)             // initializing the base class constructor with the dimension
        {}

        // The following function returns the value of the boundary force (value) at a given point which is zero
        template <int dim>
        double BoundaryForce<dim>::value(const Point<dim>&, const unsigned int) const
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
        void compute_dirichlet_constraints(const double displacment);
        void assemble_newton_system(const TrilinosWrappers::MPI::Vector& linearization_point, const bool rhs_only = false);
        void solve_newton_system(TrilinosWrappers::MPI::Vector &newton_increment);
        void solve_newton();
        void refine_grid();
        void move_mesh(const TrilinosWrappers::MPI::Vector& displacement) const;
        void output_results(const unsigned int current_refinement_cycle, const std::string output_name = "solution");

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

        // NOTE: This should be a Trillinos vector
        Vector<double> accumulated_plastic_strain;

        std::vector<SymmetricTensor<2, dim>> stress_at_q_points;

        TrilinosWrappers::SparseMatrix newton_matrix;

        TrilinosWrappers::MPI::Vector solution;
        TrilinosWrappers::MPI::Vector newton_rhs;
        TrilinosWrappers::MPI::Vector newton_rhs_uncondensed;
        TrilinosWrappers::MPI::Vector diag_mass_matrix_vector;
        TrilinosWrappers::MPI::Vector stress_tensor_diagonal;
        TrilinosWrappers::MPI::Vector stress_tensor_off_diagonal;
        TrilinosWrappers::MPI::Vector stress_tensor_tmp;
        TrilinosWrappers::MPI::Vector accumulated_plastic_strain_vector;


        const double sigma_0, hardening_slope, kappa, mu;
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
        const unsigned int n_time_steps;
        unsigned int current_step;
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
            "number of time-steps",
            "5",
            Patterns::Integer(),
            "Number of psuedo time-steps.");
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
            "Select the yield-criteria: 'Von-Mises' or 'Tresca'.");;
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

          , hardening_slope(1550.0)
          , kappa(166670.0)
          , mu(76920.0)
          , sigma_0(400.0)
          , constitutive_law(sigma_0, hardening_slope, kappa, mu)

          , base_mesh(prm.get("base mesh"))

          , applied_displacement(prm.get_double("applied displacement"))

          , yield_criteria(prm.get("yield-criteria"))

          , transfer_solution(prm.get_bool("transfer solution"))
          , n_time_steps(prm.get_integer("number of time-steps"))
          , current_step(0)

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


    template <int dim>
    void PlasticityProblem<dim>::make_grid()
    {
        // unit cube mesh
        const Point<dim> p1(0., 0., 0.);
        const Point<dim> p2(1.0, 1.0, 1.0);

        GridGenerator::hyper_rectangle(triangulation, p1, p2, true);

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
        // // setup hanging nodes and Dirichlet constraints
        // {
        //     TimerOutput::Scope t(computing_timer, "Setup: constraints");
        //     constraints_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
        //     scalar_constraints.reinit(locally_owned_scalar_dofs, locally_relevant_scalar_dofs);
        //     DoFTools::make_hanging_node_constraints(dof_handler, constraints_hanging_nodes);
        //
        //     pcout << "   Number of active cells: "
        //         << triangulation.n_global_active_cells() << std::endl
        //         << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        //         << std::endl;
        //
        //     compute_dirichlet_constraints();
        //
        //     all_constraints.copy_from(constraints_dirichlet_and_hanging_nodes);
        //     all_constraints.close();
        //     scalar_constraints.close();
        //
        //     constraints_hanging_nodes.close();
        //     constraints_dirichlet_and_hanging_nodes.close();
        // }
        // Initialization of the vectors and the active set
        {
            TimerOutput::Scope t(computing_timer, "Setup: vectors");
            solution.reinit(locally_relevant_dofs, mpi_communicator);
            newton_rhs.reinit(locally_owned_dofs, mpi_communicator);
            newton_rhs_uncondensed.reinit(locally_owned_dofs, mpi_communicator);
            diag_mass_matrix_vector.reinit(locally_owned_dofs, mpi_communicator);
            accumulated_plastic_strain.reinit(triangulation.n_active_cells());
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

        // Initialize the quadrature point history
        quadrature_point_history.resize(triangulation.n_locally_owned_active_cells() * quadrature_formula.size());

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
    }


    // How this function works will probably have to be adjusted for the course
    template <int dim>
    void PlasticityProblem<dim>::compute_dirichlet_constraints(const double displacement)
    {
        constraints_dirichlet_and_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
        constraints_dirichlet_and_hanging_nodes.merge(constraints_hanging_nodes);

        const FEValuesExtractors::Scalar x_displacement(0);
        const FEValuesExtractors::Scalar y_displacement(1);
        const FEValuesExtractors::Scalar z_displacement(2);

        // Uniaxial displacement

        VectorTools::interpolate_boundary_values(
            dof_handler,
            // top face
            5,
            // EquationData::BoundaryValues<dim>(),
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

        if (dim == 3)  // the front and back faces only exist in 3D
        {
            VectorTools::interpolate_boundary_values(
                dof_handler,
                // back face
                2,
                Functions::ZeroFunction<dim>(dim),
                constraints_dirichlet_and_hanging_nodes,
                fe.component_mask(y_displacement));
        }

        // Simple shear

        // VectorTools::interpolate_boundary_values(
        //     dof_handler,
        //     // top face
        //     5,
        //     // EquationData::BoundaryValues<dim>(),
        //     Functions::ConstantFunction<dim>(applied_displacement, dim),
        //     constraints_dirichlet_and_hanging_nodes,
        //     fe.component_mask(x_displacement));
        //
        // VectorTools::interpolate_boundary_values(
        //     dof_handler,
        //     // bottom face
        //     0,
        //     Functions::ZeroFunction<dim>(dim),
        //     constraints_dirichlet_and_hanging_nodes,
        //     fe.component_mask(x_displacement));
        //
        // VectorTools::interpolate_boundary_values(
        //     dof_handler,
        //     // left face
        //     2,
        //     Functions::ZeroFunction<dim>(dim),
        //     constraints_dirichlet_and_hanging_nodes,
        //     fe.component_mask(y_displacement));
    }


    template <int dim>
    void PlasticityProblem<dim>::assemble_newton_system(const TrilinosWrappers::MPI::Vector &linearization_point,
        const bool rhs_only)
    {
        TimerOutput::Scope t(computing_timer, "Assembling");

        const QGauss<dim> quadrature_formula(fe_degree + 1);  // Use fe_degree for quadrature formula
        const QGauss<dim - 1> face_quadrature_formula(fe_degree + 1);

        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients | update_JxW_values);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);  // element contribution to the stiffness matrix
        Vector<double> cell_rhs(dofs_per_cell);  // the element contribution to the residual

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);

        const FEValuesExtractors::Vector displacement(0);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                cell_matrix = 0.;
                cell_rhs = 0.;

                // Get strains at each quadrature point
                fe_values[displacement].get_function_symmetric_gradients(linearization_point, strain_tensor);
                // linearization_point is the solution vector from the previous iteration

                // Retrieve the quadrature point history for this cell
                unsigned int cell_index = cell->active_cell_index() * n_q_points;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    std::shared_ptr<PointHistory<dim>> &qph = quadrature_point_history[cell_index + q_point];

                    // retrieving D for step iii of Box 4.2 in the textbook
                    SymmetricTensor<4, dim> D_i = qph->get_consistent_tangent_operator();

                    // Compute the consistent tangent operator using the return mapping
                    // The following is step vii (for the current step) and step ii (for the next step) in Box 4.2 in
                    //  the textbook
                    constitutive_law.return_mapping_and_derivative_stress_strain(
                    strain_tensor[q_point], qph, yield_criteria);

                    SymmetricTensor<2, dim> stress_tensor = qph->get_stress();

                    // Assemble element tangent stiffness matrix K_T = ∑ w_i * j_i * (B_i^T * D_i * B_i)
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
                        // NOTE: This does not include the external force effects
                        //  the following is the negative of the internal force as seen in step iv in Box 4.2 in the textbook
                        // TODO: Add the external force effects to create the full residual
                        cell_rhs(i) -= transpose(fe_values[displacement].symmetric_gradient(i, q_point)) *
                            stress_tensor * fe_values.JxW(q_point);
                    }
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

        // Solve the linear system to find the Newton increment
        {
            TimerOutput::Scope t(computing_timer, "Solve: iterate");

            TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

            const double relative_accuracy = 1e-8;
            // const double solver_tolerance = relative_accuracy * newton_rhs.l2_norm();
            const double solver_tolerance = relative_accuracy * newton_matrix.residual(tmp, newton_increment, newton_rhs);

            SolverControl solver_control(3 * newton_matrix.m(), solver_tolerance);

            SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

            std::cout << "      Matrix norm      " << newton_matrix.frobenius_norm() << std::endl;
            std::cout << "      RHS norm         " << newton_rhs.l2_norm() << std::endl;

            // Solve the system: newton_matrix * newton_increment = newton_rhs
            solver.solve(newton_matrix,
                         newton_increment,
                         newton_rhs,
                         preconditioner);

            pcout << "         Error: " << solver_control.initial_value() << " -> "
                  << solver_control.last_value() << " in "
                  << solver_control.last_step() << " GMRES iterations."
                  << std::endl;
        }
    }


    template <int dim>
    void PlasticityProblem<dim>::solve_newton()
    {
        TrilinosWrappers::MPI::Vector old_solution(locally_owned_dofs, mpi_communicator);
        TrilinosWrappers::MPI::Vector r(locally_owned_dofs, mpi_communicator);  // residual vector
        TrilinosWrappers::MPI::Vector tmp_vector(locally_owned_dofs, mpi_communicator);
        TrilinosWrappers::MPI::Vector locally_relevant_tmp_vector(locally_relevant_dofs, mpi_communicator);
        TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);

        double residual_norm;

        const double tolerance = 1e-6; // Convergence tolerance for the residual norm

        double first_newton_increment_norm;

        for (unsigned int newton_step = 0; newton_step <= 100; ++newton_step)
        {
            pcout << ' ' << std::endl;
            pcout << "   Newton iteration " << newton_step << std::endl;

            double previous_residual_norm = newton_rhs.l2_norm();
            std::cout << "Previous residual norm: " << previous_residual_norm << std::endl;

            pcout << "      Assembling system... " << std::endl;
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

            assemble_newton_system(solution);  // Assemble the Newton system with the current solution

            TrilinosWrappers::MPI::Vector newton_increment(locally_owned_dofs, mpi_communicator);

            // double previous_residual_norm = newton_rhs.l2_norm();

            pcout << "      Solving system... " << std::endl;
            solve_newton_system(newton_increment);  // Solve for the Newton increment

            // double previous_residual_norm = newton_rhs.l2_norm();

            all_constraints.distribute(newton_increment);

            // Line search algorithm
            double alpha = 1.0;
            const double beta = 0.05; // Reduction factor for alpha
            const double sufficient_decrease = 1.0; // Factor for sufficient residual decrease
            TrilinosWrappers::MPI::Vector tmp_solution(locally_owned_dofs, mpi_communicator);

            while (true)
            {
                // Compute tentative solution
                tmp_solution = solution;
                tmp_solution.add(alpha, newton_increment);

                // Assemble residual with tmp_solution
                assemble_newton_system(tmp_solution, true); // Only assemble RHS (residual)

                // Compute residual norm
                r = newton_rhs;
                // const unsigned int start_res = r.local_range().first;
                // const unsigned int end_res = r.local_range().second;
                // for (unsigned int n = start_res; n < end_res; ++n)
                //     if (all_constraints.is_inhomogeneously_constrained(n))
                //         r(n) = 0.0;
                //
                // r.compress(VectorOperation::insert);

                residual_norm = r.l2_norm();

                pcout << "      Line search alpha: " << alpha << ", residual norm: " << residual_norm << std::endl;

                if (residual_norm <= previous_residual_norm * sufficient_decrease || newton_step == 0)
                {
                    // Accept the step if residual is sufficiently decreased
                    break;
                }
                else
                {
                    // Reduce alpha and try again
                    alpha -= beta;
                    if (alpha <= 0.1)
                    {
                        pcout << "      Line search failed to find acceptable alpha." << std::endl;
                        break;
                    }
                }
            }

            // Update solution with the accepted alpha
            solution = tmp_solution;

            // Update previous residual norm
            previous_residual_norm = residual_norm;

            // output_results(newton_step);

            // Check for convergence using the norm of the Newton increment
            if (newton_step == 0)
            {
                first_newton_increment_norm = newton_increment.l2_norm();

                pcout << "      First Newton increment norm: " << first_newton_increment_norm << std::endl;

                pcout << "      Residual norm: " << residual_norm << std::endl;
            }
            else
            {
                pcout << "      First Newton increment norm: " << first_newton_increment_norm << std::endl;

                pcout << "      Current Newton increment norm: " << newton_increment.l2_norm() << std::endl;

                pcout << "      Current increment norm / First increment norm: "  << newton_increment.l2_norm() / first_newton_increment_norm << std::endl;

                if (std::abs(newton_increment.l2_norm() / first_newton_increment_norm) < tolerance)
                {
                    break;
                }
            }
        }
    }


    // template <int dim>
    // void PlasticityProblem<dim>::solve_newton()
    // {
    //     TrilinosWrappers::MPI::Vector old_solution(locally_owned_dofs, mpi_communicator);
    //     TrilinosWrappers::MPI::Vector r(locally_owned_dofs, mpi_communicator);  // residual vector
    //     TrilinosWrappers::MPI::Vector tmp_vector(locally_owned_dofs, mpi_communicator);
    //     TrilinosWrappers::MPI::Vector locally_relevant_tmp_vector(locally_relevant_dofs, mpi_communicator);
    //     TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
    //
    //     double residual_norm;
    //     double previous_residual_norm;
    //
    //     const double tolerance = 1e-6; // Convergence tolerance for the residual norm
    //
    //     double first_newton_increment_norm;
    //
    //     for (unsigned int newton_step = 0; newton_step <= 100; ++newton_step)
    //     {
    //         pcout << ' ' << std::endl;
    //         pcout << "   Newton iteration " << newton_step << std::endl;
    //
    //         pcout << "      Assembling system... " << std::endl;
    //         newton_matrix = 0.;
    //         newton_rhs = 0.;
    //
    //         if (newton_step != 0)
    //         {
    //             for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
    //             {
    //                 if (all_constraints.is_inhomogeneously_constrained(n))
    //                 {
    //                     all_constraints.set_inhomogeneity(n, 0);
    //                 }
    //             }
    //         }
    //
    //         assemble_newton_system(solution);  // guess of the displacement from step k (step ii, iii and first half iv in Box 4.2 of the textbook)
    //                                            // 'solution' is the current guess of the solution
    //
    //         TrilinosWrappers::MPI::Vector newton_increment(locally_owned_dofs, mpi_communicator);
    //
    //
    //         pcout << "      Solving system... " << std::endl;
    //         solve_newton_system(newton_increment);  // solve the linear system to find the Newton increment
    //                                                    // second half of step iv in Box 4.2 of the textbook
    //
    //         all_constraints.distribute(newton_increment);
    //
    //         // NOTE: Might want to implement line search algorithm
    //         // the following is to damp the increment
    //         if (newton_step != 0)
    //         {
    //             newton_increment *= 0.3;
    //         }
    //
    //         r = newton_rhs;
    //         const unsigned int start_res = (r.local_range().first),
    //                            end_res = (r.local_range().second);
    //         for (unsigned int n = start_res; n < end_res; ++n)
    //             if (all_constraints.is_inhomogeneously_constrained(n))
    //                 r(n) = 0.0;
    //
    //         r.compress(VectorOperation::insert);
    //
    //         residual_norm = r.l2_norm();
    //
    //         output_results(newton_step);
    //
    //         // Step x: Check for convergence using the ratio of previous residual norm to current residual norm
    //         if (newton_step == 0)
    //         {
    //             first_newton_increment_norm = newton_increment.l2_norm();
    //
    //             pcout << "      First Newton increment norm: " << first_newton_increment_norm << std::endl;
    //
    //             pcout << "      Residual norm: " << residual_norm << std::endl;
    //         }
    //         else
    //         {
    //             pcout << "      First Newton increment norm: " << first_newton_increment_norm << std::endl;
    //
    //             pcout << "      Current Newton increment norm: " << newton_increment.l2_norm() << std::endl;
    //
    //             pcout << "      Current increment norm / First increment norm: "  << newton_increment.l2_norm() / first_newton_increment_norm << std::endl;
    //
    //             if (std::abs(newton_increment.l2_norm() / first_newton_increment_norm) < tolerance)
    //             {
    //                 break;
    //             }
    //         }
    //
    //         solution += newton_increment;
    //     }
    // }


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
    void PlasticityProblem<dim>::output_results(const unsigned int current_refinement_cycle,
        const std::string output_name)
    {
        TimerOutput::Scope t(computing_timer, "Graphical output");
        pcout << "      Writing graphical output... " << std::flush;

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

        accumulated_plastic_strain_vector.reinit(locally_owned_scalar_dofs, mpi_communicator);

        VectorTools::project(mapping, dof_handler_scalar, scalar_constraints, quadrature_formula,
                                     [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         const unsigned int q_point) -> double
                                     {
                                         unsigned int local_index = cell->active_cell_index() *
                                             quadrature_formula.size();
                                         const std::shared_ptr<PointHistory<dim>> &lqph =
                                             quadrature_point_history[local_index + q_point];
                                         //const SymmetricTensor<2, dim> &T = lqph->get_a();
                                         return lqph->get_accumulated_plastic_strain();
                                         // return T[i][j];
                                     },
                                     accumulated_plastic_strain_vector);
        data_out.add_data_vector(dof_handler_scalar, accumulated_plastic_strain_vector, "plastic_strain");

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

        data_out.build_patches();
        const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(output_dir,
            output_name, current_refinement_cycle, mpi_communicator, 2);
        pcout << pvtu_filename << std::endl;

        // Move mesh back
        TrilinosWrappers::MPI::Vector tmp(solution);
        tmp *= -1.0;
        move_mesh(tmp);
    }


    // The following function orchestrates the entire simulation. It manages mesh refinement, system solving
    // and output generation.
    template <int dim>
    void PlasticityProblem<dim>::run()
    {
        computing_timer.reset();

        make_grid();
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

        double displacement = 0;

        double delta_displacement = 0;

        bool reverse_loading = false;

        unsigned int n_t_steps = n_time_steps;
        const double delta_t = 1.0 / n_t_steps;

        for (unsigned int t_step = 0; t_step < n_t_steps; ++t_step)
        {
            std::cout << "Step: " << t_step << std::endl;

            if (reverse_loading == false && displacement >= applied_displacement)
            {
                reverse_loading = true;
            }

            if (reverse_loading == false)
            {
                displacement +=  t_step * delta_t * applied_displacement;
            }
            else
            {
                displacement -= t_step * delta_t * applied_displacement;
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

                compute_dirichlet_constraints(delta_displacement);

                all_constraints.copy_from(constraints_dirichlet_and_hanging_nodes);
                all_constraints.close();
                scalar_constraints.close();

                constraints_hanging_nodes.close();
                constraints_dirichlet_and_hanging_nodes.close();
            }

            solve_newton();

            output_results(t_step);
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
