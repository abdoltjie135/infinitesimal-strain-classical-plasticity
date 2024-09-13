/* Author: Abdul Gafeeth Benjamin
 * Date: 23/08/2024
 * Location: Cape Town, South Africa
 * Description: This file contains the implementation of the plasticity model
 *              and is for partial completion of MEC 5068Z
 */

// THIS IS A TEST

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

#include <sys/stat.h>
#include <cerrno>


namespace PlasticityModel
{
    using namespace dealii;

    template <int dim>
    class ConstitutiveLaw
    {
    public:
        ConstitutiveLaw(const double E, // Young's modulus
                        const double nu, // Poisson's ratio
                        const double sigma_0, // initial yield stress
                        const double gamma); // hardening parameter

        void set_sigma_0(double sigma_zero); // set the initial yield stress

        // Two functions (with parameters) being declared below
        bool get_stress_strain_tensor(
            const SymmetricTensor<2, dim>& strain_tensor,
            SymmetricTensor<4, dim>& stress_strain_tensor) const;

        void get_linearized_stress_strain_tensors(
            const SymmetricTensor<2, dim>& strain_tensor,
            SymmetricTensor<4, dim>& stress_strain_tensor_linearized,
            SymmetricTensor<4, dim>& stress_strain_tensor) const;

    private:
        const double kappa;
        const double mu;
        double sigma_0; // this has not been made constant because it will be adjusted later
        const double gamma;

        const SymmetricTensor<4, dim> stress_strain_tensor_kappa;
        const SymmetricTensor<4, dim> stress_strain_tensor_mu;
    };

    // The following is the definition of the constructor
    template <int dim>
    ConstitutiveLaw<dim>::ConstitutiveLaw(double E,
                                          double nu,
                                          double sigma_0,
                                          double gamma)
    // initialize the member variables
        : kappa(E / (3 * (1 - 2 * nu)))
          , mu(E / (2 * (1 + nu)))
          , sigma_0(sigma_0)
          , gamma(gamma)
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


    // The following function calculates the stress-strain tensor based on the strain tensor and
    // returns whether the material has yielded or not (true if it has yielded, false otherwise)
    template <int dim>
    bool ConstitutiveLaw<dim>::get_stress_strain_tensor(
        const SymmetricTensor<2, dim>& strain_tensor,
        SymmetricTensor<4, dim>& stress_strain_tensor) const
    {
        Assert(dim == 3, ExcNotImplemented());

        SymmetricTensor<2, dim> stress_tensor;
        stress_tensor =
            (stress_strain_tensor_kappa + stress_strain_tensor_mu) * strain_tensor;

        const SymmetricTensor<2, dim> deviator_stress_tensor = deviator(stress_tensor);
        const double deviator_stress_tensor_norm = deviator_stress_tensor.norm();

        stress_strain_tensor = stress_strain_tensor_mu;
        if (deviator_stress_tensor_norm > sigma_0)
        {
            const double beta = sigma_0 / deviator_stress_tensor_norm;
            stress_strain_tensor *= (gamma + (1 - gamma) * beta);
        }

        stress_strain_tensor += stress_strain_tensor_kappa;

        return (deviator_stress_tensor_norm > sigma_0); // checking if yielding has occurred
    }


    // A large part of the function below is the same as the get_stress_strain_tensor.
    // The following function also computes a linearized_stress_strain tensor which is necessary while
    // nonlinear problems. The Newton-Raphson method requires linearizing the nonlinear equations.
    template <int dim>
    void ConstitutiveLaw<dim>::get_linearized_stress_strain_tensors(
        const SymmetricTensor<2, dim>& strain_tensor,
        SymmetricTensor<4, dim>& stress_strain_tensor_linearized,
        SymmetricTensor<4, dim>& stress_strain_tensor) const
    {
        Assert(dim == 3, ExcNotImplemented()); // checking if the code is being run in 3D

        SymmetricTensor<2, dim> stress_tensor;
        stress_tensor = (stress_strain_tensor_kappa + stress_strain_tensor_mu) * strain_tensor;

        stress_strain_tensor = stress_strain_tensor_mu;
        stress_strain_tensor_linearized = stress_strain_tensor_mu;

        SymmetricTensor<2, dim> deviator_stress_tensor = deviator(stress_tensor);
        const double deviator_stress_tensor_norm = deviator_stress_tensor.norm();

        if (deviator_stress_tensor_norm > sigma_0)
        {
            const double beta = sigma_0 / deviator_stress_tensor_norm;
            stress_strain_tensor *= (gamma + (1 - gamma) * beta);
            stress_strain_tensor_linearized *= (gamma + (1 - gamma) * beta);
            deviator_stress_tensor /= deviator_stress_tensor_norm;
            stress_strain_tensor_linearized -=
                (1 - gamma) * beta * 2 * mu * outer_product(deviator_stress_tensor, deviator_stress_tensor);
        }

        stress_strain_tensor += stress_strain_tensor_kappa;
        stress_strain_tensor_linearized += stress_strain_tensor_kappa;
    }


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
        BoundaryForce<dim>::BoundaryForce() // constructor definition
            : Function<dim>(dim) // initializing the base class constructor with the dimension
        {
        }

        // The following function returns the value of the boundary force (value) at a given point which is zero
        template <int dim>
        double BoundaryForce<dim>::value(const Point<dim>&, const unsigned int) const // parameters are not
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
        PlasticityProblem(const ParameterHandler& prm); // constructor which initializes an instance of the class
        // with the parameters in the prm object

        void run(); // function to run the simulation

        static void declare_parameters(ParameterHandler& prm); // function to declare the parameters

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
        parallel::distributed::Triangulation<dim> triangulation; // distributing the mesh across multiple processors
        // for parallel computing

        const unsigned int fe_degree;
        const FESystem<dim> fe;
        DoFHandler<dim> dof_handler;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        AffineConstraints<double> constraints_hanging_nodes;
        AffineConstraints<double> constraints_dirichlet_and_hanging_nodes;
        AffineConstraints<double> all_constraints;

        IndexSet active_set; // not sure if this is needed for the plasticity code
        Vector<float> fraction_of_plastic_q_points_per_cell;

        TrilinosWrappers::SparseMatrix newton_matrix;

        TrilinosWrappers::MPI::Vector solution;
        TrilinosWrappers::MPI::Vector newton_rhs;
        TrilinosWrappers::MPI::Vector newton_rhs_uncondensed;
        TrilinosWrappers::MPI::Vector diag_mass_matrix_vector;

        const double e_modulus, nu, gamma, sigma_0;
        ConstitutiveLaw<dim> constitutive_law;

        const std::string base_mesh;

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
    }


    template <int dim>
    PlasticityProblem<dim>::PlasticityProblem(const ParameterHandler& prm) // defining the constructor
    // PlasticityProblem
        : mpi_communicator(MPI_COMM_WORLD)
          , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          // the following initializes a timer to track computation times
          , computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::never, TimerOutput::wall_times)

          , n_initial_global_refinements(prm.get_integer("number of initial refinements"))
          , triangulation(mpi_communicator)
          , fe_degree(prm.get_integer("polynomial degree"))
          , fe(FE_Q<dim>(QGaussLobatto<1>(fe_degree + 1)) ^ dim)
          , dof_handler(triangulation)

          , e_modulus(200000)
          , nu(0.3)
          , gamma(0.01)
          , sigma_0(400.0)
          , constitutive_law(e_modulus, nu, sigma_0, gamma)

          , base_mesh(prm.get("base mesh"))

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

            // default: 0 left had face, 1 right hand face, 2 bottom face, 3 top face, 4 front face, 5 back face

            /*
            // the following assigns boundary IDs to the faces of the mesh
            // inequality symbols are used because of floating-point arithmetic
            for (const auto& cell : triangulation.active_cell_iterators())
                for (const auto& face : cell->face_iterators()) // checks if the face is at the boundary
                    if (face->at_boundary())
                    {
                        // top face
                        if (std::fabs(face->center()[2] - p2[2]) < 1e-12) // face at z=1
                            face->set_boundary_id(1);
                        // side faces
                        if (std::fabs(face->center()[0] - p1[0]) < 1e-12 || // face at x=0
                            std::fabs(face->center()[0] - p2[0]) < 1e-12 || // face at x=1
                            std::fabs(face->center()[1] - p1[1]) < 1e-12 || // face at y=0
                            std::fabs(face->center()[1] - p2[1]) < 1e-12) // face at y=1
                            face->set_boundary_id(8);
                        // bottom face
                        if (std::fabs(face->center()[2] - p1[2]) < 1e-12) // face at z=0
                            face->set_boundary_id(6);
                    }
              */
        }

        triangulation.refine_global(n_initial_global_refinements); // refine the mesh globally based on prm file
    }


    template <int dim>
    void PlasticityProblem<dim>::setup_system()
    {
        // setup dofs and get index sets for locally owned and relevant DoFs
        {
            TimerOutput::Scope t(computing_timer, "Setup: distribute DoFs");
            dof_handler.distribute_dofs(fe);

            locally_owned_dofs = dof_handler.locally_owned_dofs();
            locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
            // DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
        }

        // setup hanging nodes and Dirichlet constraints
        {
            TimerOutput::Scope t(computing_timer, "Setup: constraints");
            constraints_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
            // constraints_dirichlet_and_hanging_nodes.reinit(locally_relevant_dofs);
            DoFTools::make_hanging_node_constraints(dof_handler, constraints_hanging_nodes);


            pcout << "   Number of active cells: "
                << triangulation.n_global_active_cells() << std::endl
                << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

            compute_dirichlet_constraints();

            all_constraints.copy_from(constraints_dirichlet_and_hanging_nodes);
            all_constraints.close();

            constraints_hanging_nodes.close();
            constraints_dirichlet_and_hanging_nodes.close();
            all_constraints.close();
        }

        // Initialization of the vectors and the active set
        {
            TimerOutput::Scope t(computing_timer, "Setup: vectors");
            solution.reinit(locally_relevant_dofs, mpi_communicator);
            newton_rhs.reinit(locally_owned_dofs, mpi_communicator);
            newton_rhs_uncondensed.reinit(locally_owned_dofs, mpi_communicator);
            diag_mass_matrix_vector.reinit(locally_owned_dofs, mpi_communicator);
            fraction_of_plastic_q_points_per_cell.reinit(triangulation.n_active_cells());
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
    }


    // How this function works will probably have to be adjusted for the course
    template <int dim>
    void PlasticityProblem<dim>::compute_dirichlet_constraints()
    {
        constraints_dirichlet_and_hanging_nodes.reinit(locally_owned_dofs, locally_relevant_dofs);
        // constraints_dirichlet_and_hanging_nodes.reinit(locally_relevant_dofs);
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
                3,
                // EquationData::BoundaryValues<dim>(),
                Functions::ConstantFunction<dim>(0.4, dim),
                constraints_dirichlet_and_hanging_nodes,
                fe.component_mask(z_displacement));

            VectorTools::interpolate_boundary_values(
                dof_handler,
                // bottom face
                2,
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

            if (dim == 3)
            {
                VectorTools::interpolate_boundary_values(
                    dof_handler,
                    // front face
                    4,
                    Functions::ZeroFunction<dim>(dim),
                    constraints_dirichlet_and_hanging_nodes,
                    fe.component_mask(y_displacement));
            }

            /*VectorTools::interpolate_boundary_values(
                dof_handler,
                // the sides of the box
                8,
                EquationData::BoundaryValues<dim>(),
                constraints_dirichlet_and_hanging_nodes,
                (fe.component_mask(x_displacement) | fe.component_mask(y_displacement)));*/
        }
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
    void PlasticityProblem<dim>::assemble_newton_system(
        const TrilinosWrappers::MPI::Vector& linearization_point)
    {
        TimerOutput::Scope t(computing_timer, "Assembling");

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
        std::vector<Vector<double>> boundary_force_values(n_face_q_points,
                                                          Vector<double>(dim));

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);  // initialize the local dof indices

        const FEValuesExtractors::Vector displacement(0);

        for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                cell_matrix = 0;
                cell_rhs = 0;

                std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);
                fe_values[displacement].get_function_symmetric_gradients(
                    linearization_point, strain_tensor);

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    SymmetricTensor<4, dim> stress_strain_tensor_linearized;
                    SymmetricTensor<4, dim> stress_strain_tensor;
                    constitutive_law.get_linearized_stress_strain_tensors(
                        strain_tensor[q_point],
                        stress_strain_tensor_linearized,
                        stress_strain_tensor);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const SymmetricTensor<2, dim> stress_phi_i =
                            stress_strain_tensor_linearized *
                            fe_values[displacement].symmetric_gradient(i, q_point);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            cell_matrix(i, j) +=
                            (stress_phi_i *
                                fe_values[displacement].symmetric_gradient(j, q_point) *
                                fe_values.JxW(q_point));
                        }

                        cell_rhs(i) +=
                        ((stress_phi_i -
                                stress_strain_tensor *
                                fe_values[displacement].symmetric_gradient(i,
                                                                           q_point)) *
                            strain_tensor[q_point] * fe_values.JxW(q_point));
                    }
                }

                cell->get_dof_indices(local_dof_indices);

                // attempt to distribute local to global
                try {
                    all_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                        newton_matrix, newton_rhs, true);
                } catch (std::exception &e) {
                    std::cerr << "Exception during distribute_local_to_global: " << e.what() << std::endl;
                    throw;
                }

                all_constraints.distribute_local_to_global(cell_matrix,
                                                           cell_rhs,
                                                           local_dof_indices,
                                                           newton_matrix,
                                                           newton_rhs,
                                                           true);
            }

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
                        strain_tensors[q_point], stress_strain_tensor);
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

        // move the mesh to reflect the current deformation
        move_mesh(solution);

        // Initialize DataOut object for output
        DataOut<dim> data_out;

        // Attach the DoF handler to the DataOut object
        data_out.attach_dof_handler(dof_handler);

        // add displacement to the data output
        const std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_out.add_data_vector(solution,
                                 std::vector<std::string>(dim, "displacement"),
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);
        // add the fraction of plastic quadrature points per cell to the data output
        data_out.add_data_vector(fraction_of_plastic_q_points_per_cell,
                                 "fraction_of_plastic_q_points");

        // add subdomain information to the data output
        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "subdomain");

        // Build patches and write output files
        data_out.build_patches();
        const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
            output_dir, "solution", current_refinement_cycle,
            mpi_communicator, 2);
        pcout << pvtu_filename << std::endl;

        // move the mesh back to its original position
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

            output_results(current_refinement_cycle);

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
            std::cerr << "*** Call this program as <./plasticity_model test.prm>"
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
