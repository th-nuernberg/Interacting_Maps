//
// Created by root on 7/29/25.
//
#include <update.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS HELPER FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool isApprox(Tensor3f &t1, Tensor3f &t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

bool isApprox(Tensor2f &t1, Tensor2f &t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

void norm_tensor_along_dim3(const Tensor3f &T, Tensor2f &norm){
    array<int,1> dims({2});
    norm = T.square().sum(dims).sqrt();
}

autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, float fx, float fy, float cx, float cy) {
    autodiff::Vector3real result;
    // Camera frame: x right, y down, z forward (right-handed). Stored as (y, x, z).
    result << (y - cy) / fy,
              (x - cx) / fx,
              1;
    return result;
}

autodiff::Vector3real C(autodiff::real x, autodiff::real y, float fx, float fy, float cx, float cy) {
    autodiff::Vector3real c_star = C_star(x, y, fx, fy, cx, cy);
    autodiff::real norm = sqrt(c_star.squaredNorm());
    return c_star / norm;
}

void find_C(int N_x, int N_y, float fx, float fy, float cx, float cy, Tensor3f &CCM, Tensor3f &C_x, Tensor3f &C_y) {
    MatrixXf XX(N_y, N_x);
    MatrixXf YY(N_y, N_x);
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            XX(i, j) = float(j);
            YY(i, j) = float(i);
        }
    }
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            autodiff::real x = XX(i, j);
            autodiff::real y = YY(i, j);

            // Compute the function value
            autodiff::Vector3real c_val = C(x, y, fx, fy, cx, cy);
            CCM(i,j,0) = static_cast<float>(c_val(0)); // y
            CCM(i,j,1) = static_cast<float>(c_val(1)); // x
            CCM(i,j,2) = static_cast<float>(c_val(2)); // z
            // Compute the jacobians
            autodiff::VectorXreal F;

            // NEEDS TO STAY D O U B L E
            VectorXd dCdx;
            autodiff::jacobian(C, wrt(x), at(x, y, fx, fy, cx, cy), F, dCdx);
            VectorXd dCdy;
            autodiff::jacobian(C, wrt(y), at(x, y, fx, fy, cx, cy), F, dCdy);

            // C_x = dCdx
            C_x(i,j,0) = (float) dCdx(0); // y
            C_x(i,j,1) = (float) dCdx(1); // x
            C_x(i,j,2) = (float) dCdx(2); // z

            // C_y = dCdy (y and rows both point down)
            C_y(i,j,0) = (float) dCdy(0); // y
            C_y(i,j,1) = (float) dCdy(1); // x
            C_y(i,j,2) = (float) dCdy(2); // z
        }
    }
}

void crossProduct3x3(const Tensor3f &A, const Tensor3f &B, Tensor3f &C) {
    const auto& dims = A.dimensions();
    long rows = dims[0]; // height
    long cols = dims[1]; // width
    for (long i = 0; i < rows; ++i){
        for (long j = 0; j < cols; ++j){
            C(i, j, 0) = A(i, j, 2) * B(i, j, 1) - A(i, j, 1) * B(i, j, 2);  // y
            C(i, j, 1) = A(i, j, 0) * B(i, j, 2) - A(i, j, 2) * B(i, j, 0);  // x
            C(i, j, 2) = A(i, j, 1) * B(i, j, 0) - A(i, j, 0) * B(i, j, 1);  // z
        }
    }
}

void crossProduct3x3(const Tensor3f &A, const Vector3f &B, Vector3f &C, int y, int x) {
    C(0) = A(y, x, 2) * B(1) - A(y, x, 1) * B(2);  // y
    C(1) = A(y, x, 0) * B(2) - A(y, x, 2) * B(0);  // x
    C(2) = A(y, x, 1) * B(0) - A(y, x, 0) * B(1);  // z
}

void crossProduct1x3(const Tensor1f &A, const Tensor3f &B, Tensor3f &C){
    const auto& dimensions = B.dimensions();
    for (long i = 0; i < dimensions[0]; ++i){
        for (long j = 0; j < dimensions[1]; ++j){
            C(i, j, 0) = A(2) * B(i, j, 1) - A(1) * B(i, j, 2);  // y
            C(i, j, 1) = A(0) * B(i, j, 2) - A(2) * B(i, j, 0);  // x
            C(i, j, 2) = A(1) * B(i, j, 0) - A(0) * B(i, j, 1);  // z
        }
    }

}

void vector_distance(const Tensor3f &vec1, const Tensor3f &vec2, Tensor2f &distance){
    PROFILE_FUNCTION();
    const auto& dimensions = vec1.dimensions();
    Tensor3f cross_product(dimensions);
    Tensor2f norm(dimensions[0], dimensions[1]);
    Tensor2f norm2(dimensions[0], dimensions[1]);
    crossProduct3x3(vec1, vec2, cross_product);
    norm.setZero();
    norm2.setZero();
    norm_tensor_along_dim3(cross_product, norm);
    norm_tensor_along_dim3(vec2, norm2);
    distance = norm/norm2;
}

float sign_func(float x){
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}

void computeDotProductWithLoops(const Tensor3f &A, const Tensor3f &B, Tensor2f &D) {
    PROFILE_FUNCTION();
    const int height = (int) A.dimension(0);
    const int width = (int) A.dimension(1);
    const int depth = (int) A.dimension(2);

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float dotProduct = 0.0f; // Initialize the dot product for position (i, j)
            for (int k = 0; k < depth; ++k) {
                dotProduct += A(i, j, k) * B(i, j, k);
            }
            D(i, j) = dotProduct; // Store the result in tensor D
        }
    }
}

void m32(const Tensor3f &In, const Tensor3f &C_x, const Tensor3f &C_y, Tensor3f &Out){
    const auto& dimensions = In.dimensions();
    Tensor3f C1(dimensions);
    Tensor3f C2(dimensions);
    Tensor2f dot(dimensions[0], dimensions[1]);
    Tensor2f sign(dimensions[0], dimensions[1]);
    Tensor2f distance1(dimensions[0], dimensions[1]);
    Tensor2f distance2(dimensions[0], dimensions[1]);

    crossProduct3x3(C_x,C_y,C1);
    crossProduct3x3(C_y,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::function(sign_func));
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    Out.chip(1,2) = sign * distance1/distance2;

    crossProduct3x3(C_y,C_x,C1);
    crossProduct3x3(C_x,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::function(sign_func));
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    Out.chip(0,2) = sign * distance1/distance2;
}

void m23(const Tensor3f &In, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &Out, int y, int x) {
    Out(0) = In(y, x, 1) * Cx(y, x, 0) + In(y, x, 0) * Cy(y, x, 0);
    Out(1) = In(y, x, 1) * Cx(y, x, 1) + In(y, x, 0) * Cy(y, x, 1);
    Out(2) = In(y, x, 1) * Cx(y, x, 2) + In(y, x, 0) * Cy(y, x, 2);
}

void computeGradient(const Tensor2f &data, Tensor3f &gradients, int y, int x) {
    PROFILE_FUNCTION();
    // Compute gradient for update_IG
    const auto& gdimensions = data.dimensions();
    int rows = static_cast<int>(gdimensions[0]);
    int cols = static_cast<int>(gdimensions[1]);
    assert(y < rows);
    assert(x < cols);

    // Compute gradient along columns (y-direction, pointing down)
    if (y == 0) {
        gradients(y, x, 0) = (data(y + 1, x) - data(y, x)) / 2.0f; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y, x) - data(y - 1, x)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y + 1, x) - data(y - 1, x)) / 2.0f;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x)) / 2.0f; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x) - data(y, x - 1)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x - 1)) / 2.0f;
    }
}

void computeGradient(const Tensor3f &data, Tensor3f &gradients, int y, int x) {
    // Compute gradient for update_IG
    const auto& dimensions = data.dimensions();
    int rows = static_cast<int>(dimensions[0]);
    int cols = static_cast<int>(dimensions[1]);
    assert(y < rows);
    assert(x < cols);
    assert(static_cast<int>(dimensions[2]) == 2);
    // Compute gradient along columns (y-direction, pointing down)
    if (y == 0) {
        gradients(y, x, 0) = (data(y + 1, x, 0) - data(y, x, 0)) / 2.0f; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y, x, 0) - data(y - 1, x, 0)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y + 1, x, 0) - data(y - 1, x, 0)) / 2.0f;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x, 1)) / 2.0f; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x, 1) - data(y, x - 1, 1)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x - 1, 1)) / 2.0f;
    }
}

float VFG_check(const Tensor2f &V, const Tensor3f &F, const Tensor3f &G){
    const auto& dimensions = F.dimensions();
    MatrixXfRowMajor dot(dimensions[0], dimensions[1]);
    MatrixXfRowMajor diff(dimensions[0], dimensions[1]);

    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            dot(i,j) = -(F(i,j,0)*G(i,j,0) + F(i,j,1)*G(i,j,1));
            diff(i,j) = (V(i,j) - dot(i,j));
        }
    }
    return diff.lpNorm<Infinity>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS UPDATE FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void setup_R_update(const Tensor3f &CCM, Matrix3f &A, Vector3f &B, std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &points){
    PROFILE_FUNCTION();
    const auto &dimensions = CCM.dimensions();
    int rows = (int) dimensions[0];
    int cols = (int) dimensions[1];
    Matrix3f Identity = Matrix3f::Identity();
    Vector3f d;
    B.setZero();

    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j < cols; ++j){
            d(0) = CCM((int) i, (int) j, 0);
            d(1) = CCM((int) i, (int) j, 1);
            d(2) = CCM((int) i, (int) j, 2);
            Identity_minus_outerProducts[i][j] = Identity - d * d.transpose();
            A += Identity_minus_outerProducts[i][j];
            points[i][j].setZero();
        }
    }
}

void rebuild_R_sums(const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &points){
    PROFILE_FUNCTION();
    const auto &dimensions = F.dimensions();
    Vector3f transformed_F;
    B.setZero();
    for (int i = 0; i < dimensions[0]; ++i){
        for (int j = 0; j < dimensions[1]; ++j){
            m23(F, Cx, Cy, transformed_F, i, j);
            crossProduct3x3(C, transformed_F, points[i][j], i, j);
            B += Identity_minus_outerProducts[i][j] * points[i][j];
        }
    }
}

void update_FG(Tensor3f &F, const float V, const Tensor3f &G, int y, int x, const float lr, const float weight_FG, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Vector2f update_F;
    update_F.setZero();
    float norm = std::abs((G(y, x, 0) * G(y, x, 0) + G(y, x, 1) * G(y, x, 1)));
    if (norm != 0.0) {
        update_F(0) = F(y, x, 0) - ((G(y, x, 0) / norm) * (V + (F(y, x, 0) * G(y, x, 0) + F(y, x, 1) * G(y, x, 1))));
        update_F(1) = F(y, x, 1) - ((G(y, x, 1) / norm) * (V + (F(y, x, 0) * G(y, x, 0) + F(y, x, 1) * G(y, x, 1))));
        F(y, x, 0) = (1 - weight_FG) * F(y, x, 0) + lr * weight_FG * update_F(0);
        F(y, x, 1) = (1 - weight_FG) * F(y, x, 1) + lr * weight_FG * update_F(1);
        if (F(y, x, 0) > gamma){
            F(y, x, 0) = gamma;
        }
        if (F(y, x, 1) > gamma){
            F(y, x, 1) = gamma;
        }
        if (F(y, x, 0) < -gamma){
            F(y, x, 0) = -gamma;
        }
        if (F(y, x, 1) < -gamma){
            F(y, x, 1) = -gamma;
        }
        if (std::abs(F(y, x, 0)) < eps){
            F(y, x, 0) = 0.0;
        }
        if (std::abs(F(y, x, 1)) < eps){
            F(y, x, 1) = 0.0;
        }
    }
}

void update_GF(Tensor3f &G, float V, const Tensor3f &F, int y, int x, const float lr, const float weight_GF, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Vector2f update_G;
    update_G.setZero();
    float norm = std::abs((F(y, x, 0) * F(y, x, 0) + F(y, x, 1) * F(y, x, 1)));
    if (norm != 0.0) {
        update_G(0) = G(y, x, 0) - ((F(y, x, 0) / norm) * (V + (G(y, x, 0) * F(y, x, 0) + G(y, x, 1) * F(y, x, 1))));
        update_G(1) = G(y, x, 1) - ((F(y, x, 1) / norm) * (V + (G(y, x, 0) * F(y, x, 0) + G(y, x, 1) * F(y, x, 1))));
        G(y, x, 0) = (1 - weight_GF) * G(y, x, 0) + lr * weight_GF * update_G(0);
        G(y, x, 1) = (1 - weight_GF) * G(y, x, 1) + lr * weight_GF * update_G(1);
        if (G(y, x, 0) > gamma){
            G(y, x, 0) = gamma;
        }
        if (G(y, x, 1) > gamma){
            G(y, x, 1) = gamma;
        }
        if (G(y, x, 0) < -gamma){
            G(y, x, 0) = -gamma;
        }
        if (G(y, x, 1) < -gamma){
            G(y, x, 1) = -gamma;
        }
        if (std::abs(G(y, x, 0)) < eps){
            G(y, x, 0) = 0.0;
        }
        if (std::abs(G(y, x, 1)) < eps){
            G(y, x, 1) = 0.0;
        }
    }
}

void update_GI(Tensor3f &G, const Tensor3f &I_gradient, int y, int x, float weight_GI, float eps, float gamma){
    PROFILE_FUNCTION();
    G(y, x, 0) = (1 - weight_GI) * G(y, x, 0) + weight_GI*I_gradient(y, x, 0);
    G(y, x, 1) = (1 - weight_GI) * G(y, x, 1) + weight_GI*I_gradient(y, x, 1);
    if (G(y, x, 0) > gamma){
        G(y, x, 0) = gamma;
    }
    if (G(y, x, 1) > gamma){
        G(y, x, 1) = gamma;
    }
    if (G(y, x, 0) < -gamma){
        G(y, x, 0) = -gamma;
    }
    if (G(y, x, 1) < -gamma){
        G(y, x, 1) = -gamma;
    }
    if (std::abs(G(y, x, 0)) < eps){
        G(y, x, 0) = 0;
    }
    if (std::abs(G(y, x, 1)) < eps){
        G(y, x, 1) = 0;
    }
}

void updateGIDiffGradient(Tensor3f &G, Tensor3f &I_gradient, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, int y, int x){
    PROFILE_FUNCTION();
    // The divergence at (y, x) reads the residual at the pixel and its 4 neighbours, so refresh all of them.
    const int rows = static_cast<int>(GIDiff.dimension(0));
    const int cols = static_cast<int>(GIDiff.dimension(1));
    const int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto &offset : offsets) {
        const int yy = y + offset[0];
        const int xx = x + offset[1];
        if (yy < 0 || yy >= rows || xx < 0 || xx >= cols) continue;
        GIDiff(yy, xx, 0) = G(yy, xx, 0) - I_gradient(yy, xx, 0);
        GIDiff(yy, xx, 1) = G(yy, xx, 1) - I_gradient(yy, xx, 1);
    }
    computeGradient(GIDiff, GIDiffGradient, y, x);
}

void update_IG(Tensor2f &I, const Tensor3f &GIDiffGradient, int y, int x, float weight_IG){
    PROFILE_FUNCTION();
    I(y, x) = I(y, x) + weight_IG * (- GIDiffGradient(y, x, 0) - GIDiffGradient(y, x, 1));
}

void contribute(Tensor2f &I, float V, int y, int x, float minPotential, float maxPotential, const float weight_IV){
    I(y, x) = std::min(std::max(I(y, x) + weight_IV * V, minPotential), maxPotential);
}

void globalDecay(Tensor2f &I, Tensor2f &decayTimeSurface, Tensor2f &nP, Tensor2f &t, Tensor2f &dP) {
    const Tensor2f lastPotential = I;
    I = (lastPotential - nP) * (-(t - decayTimeSurface) / dP).exp() + nP;
    decayTimeSurface = t;
}

void linearDecay(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam){
    const float lastDecayTime = decayTimeSurface(y,x);
    const float lastPotential = I(y,x);
    I(y,x)
        = (lastPotential >= neutralPotential)
            ? std::max(lastPotential - (time - lastDecayTime) * decayParam, neutralPotential)
            : std::min(lastPotential + (time - lastDecayTime) * decayParam, neutralPotential);
    decayTimeSurface(y, x) = time;
}

void exponentialDecay(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam){
    const float lastDecayTime = decayTimeSurface(y,x);
    const float lastPotential = I(y,x);
    I(y,x)
        = (lastPotential - neutralPotential) * expf(-(time-lastDecayTime) / decayParam)
        + neutralPotential;
    decayTimeSurface(y,x) = time;
    //float newIntensity = (I(y, x) - neutralPotential) * expf(-(time - decayTimeSurface(y, x)) / decayParam) + neutralPotential;
    //I(y, x) = newIntensity;
    //decayTimeSurface(y, x) = time;
}

void decayNeighbourhood(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam){
    PROFILE_FUNCTION();
    // event_step takes central differences at (y, x) and its 4 neighbours, which read every pixel within
    // Manhattan distance 2. Decay is lazy, so all of them must be brought to the same time first.
    const int rows = static_cast<int>(I.dimension(0));
    const int cols = static_cast<int>(I.dimension(1));
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int yy = y + dy;
            const int xx = x + dx;
            if (std::abs(dy) + std::abs(dx) > 2 || yy < 0 || yy >= rows || xx < 0 || xx >= cols) continue;
            exponentialDecay(I, decayTimeSurface, yy, xx, time, neutralPotential, decayParam);
        }
    }
}

void relaxSilentG(Tensor3f &G, const Tensor3f &F, const Tensor3f &I_gradient, Tensor2f &GTimeSurface, const int y, const int x, const float time, const float rate_GI, const float rate_GF){
    PROFILE_FUNCTION();
    // No event at (y, x) since GTimeSurface(y, x) is evidence V = 0 for that whole time. At such a pixel the paper
    // keeps applying G <- I (pull towards g = grad I) and G <- F with V = 0 (remove G's component along F).
    // As continuous-time relaxation with g taken constant over the silent time, split along F:
    //   perpendicular: G' = -rate_GI (G - g)                    -> g + (G - g) exp(-rate_GI dt)
    //   parallel:      G' = -rate_GI (G - g) - rate_GF G        -> G* + (G - G*) exp(-(rate_GI + rate_GF) dt)
    // with G* = rate_GI / (rate_GI + rate_GF) g. F is taken as constant over the silent time.
    const float dt = time - GTimeSurface(y, x);
    GTimeSurface(y, x) = time;
    if (dt <= 0.0f) return;
    const Vector2f g(I_gradient(y, x, 0), I_gradient(y, x, 1));
    const Vector2f G_old(G(y, x, 0), G(y, x, 1));
    const Vector2f f(F(y, x, 0), F(y, x, 1));
    const float norm = f.squaredNorm();
    const Vector2f f_hat = norm > 0.0f ? Vector2f(f / std::sqrt(norm)) : Vector2f::Zero();

    const float G_par = G_old.dot(f_hat);
    const float g_par = g.dot(f_hat);
    const Vector2f G_perp = G_old - G_par * f_hat;
    const Vector2f g_perp = g - g_par * f_hat;

    const float rate_par = rate_GI + rate_GF;
    const float G_par_fixed = rate_par > 0.0f ? rate_GI / rate_par * g_par : G_par;
    const Vector2f G_perp_new = g_perp + (G_perp - g_perp) * expf(-rate_GI * dt);
    const float G_par_new = G_par_fixed + (G_par - G_par_fixed) * expf(-rate_par * dt);
    G(y, x, 0) = G_perp_new(0) + G_par_new * f_hat(0);
    G(y, x, 1) = G_perp_new(1) + G_par_new * f_hat(1);
}

void relaxSilentNeighbourhood(Tensor3f &G, const Tensor3f &F, const Tensor2f &I, Tensor3f &I_gradient, Tensor2f &GTimeSurface, const int y, const int x, const float time, const float rate_GI, const float rate_GF){
    PROFILE_FUNCTION();
    // event_step reads and writes G at (y, x) and its 4 neighbours. I must already be decayed to time
    // (decayNeighbourhood) so that grad I is current.
    const int rows = static_cast<int>(G.dimension(0));
    const int cols = static_cast<int>(G.dimension(1));
    const int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto &offset : offsets) {
        const int yy = y + offset[0];
        const int xx = x + offset[1];
        if (yy < 0 || yy >= rows || xx < 0 || xx >= cols) continue;
        computeGradient(I, I_gradient, yy, xx);
        relaxSilentG(G, F, I_gradient, GTimeSurface, yy, xx, time, rate_GI, rate_GF);
    }
}

Tensor3f relaxedSilentCopy(const Tensor3f &G, const Tensor3f &F, const Tensor2f &I, const Tensor2f &GTimeSurface, const float time, const float rate_GI, const float rate_GF){
    PROFILE_FUNCTION();
    Tensor3f relaxed = G;
    Tensor2f times = GTimeSurface;
    Tensor3f I_gradient(G.dimensions());
    for (int y = 0; y < G.dimension(0); ++y) {
        for (int x = 0; x < G.dimension(1); ++x) {
            computeGradient(I, I_gradient, y, x);
            relaxSilentG(relaxed, F, I_gradient, times, y, x, time, rate_GI, rate_GF);
        }
    }
    return relaxed;
}

Tensor2f decayedCopy(const Tensor2f &I, const Tensor2f &decayTimeSurface, const float time, const float neutralPotential, const float decayParam){
    PROFILE_FUNCTION();
    // Exponential decay composes exactly, so decaying a copy for display does not change the dynamics.
    Tensor2f decayed = I;
    Tensor2f times = decayTimeSurface;
    for (int y = 0; y < I.dimension(0); ++y) {
        for (int x = 0; x < I.dimension(1); ++x) {
            exponentialDecay(decayed, times, y, x, time, neutralPotential, decayParam);
        }
    }
    return decayed;
}

void update_IV(Tensor2f &I, const float V, const int y, const int x, const float minPotential, const float maxPotential, const float weight_IV){
    PROFILE_FUNCTION();
    contribute(I, V, y, x, minPotential, maxPotential, weight_IV);
}

void update_Ifusion(Tensor2f &I, const cv::Mat &realImage, const float weight_Ifusion) {
    Tensor2f lastPotential = I;
    I = (1-weight_Ifusion) * lastPotential + 255 * weight_Ifusion * Matrix2Tensor(cvMatToEigen(realImage));
}

void update_FR(Tensor3f &F, const Tensor3f &CCM, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, const float weight_FR, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Tensor3f cross(CCM.dimensions());
    const auto& dimensions = F.dimensions();
    Tensor3f update(F.dimensions());
    {
        PROFILE_SCOPE("FR CROSS PRODUCT");
        crossProduct1x3(R, CCM, cross);
    }
    {
        PROFILE_SCOPE("FR M32");
        m32(cross, Cx, Cy, update);
    }
    F = (1 - weight_FR)*F + weight_FR*update;
    for (int i = 0; i<dimensions[0]; i++) {
        for (int j = 0; j < dimensions[1]; j++) {
            if (F(i, j, 0) > gamma){
                F(i, j, 0) = gamma;
            }
            if (F(i, j, 1) > gamma){
                F(i, j, 1) = gamma;
            }
            if (F(i, j, 0) < -gamma){
                F(i, j, 0) = -gamma;
            }
            if (F(i, j, 1) < -gamma){
                F(i, j, 1) = -gamma;
            }
            if (std::abs(F(i,j,0)) < eps){
                F(i,j,0) = 0;
            }
            if (std::abs(F(i,j,1)) < eps){
                F(i,j,1) = 0;
            }
        }
    }
}

//void update_RF(Tensor1f &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, const float weight_RF, const std::vector<Event> &frameEvents) {
//    PROFILE_FUNCTION();
//    const auto &dimensions = F.dimensions();
//    Vector3f transformed_F(3);
//    Vector3f point(3);
//    Vector3f solution(3);
//    {
//        PROFILE_SCOPE("RF Pre");
//        for (auto event : frameEvents){
//            // Transform F from 2D image space to 3D world space with C
//            m23(F, Cx, Cy, transformed_F, event.coordinates[0], event.coordinates[1]);
//            // calculate cross product between world space F and calibration matrix.
//            // this gives us the point on which the line stands
//            crossProduct3x3(C, transformed_F, point, event.coordinates[0], event.coordinates[1]);
//            // right hand side B consists of a sum of a points
//            // subtract the contribution of the old_point at y,x and add the contribution of the new point
//            B = B - Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*old_points[event.coordinates[0]][event.coordinates[1]] + Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*point;
//            // new point is now old
//            old_points[event.coordinates[0]][event.coordinates[1]] = point;
//        }
//    }
//    // solve for the new rotation vector
//    solution = A.partialPivLu().solve(B);
//    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
//    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
//    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
//}


void update_RF(Tensor1f &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, float weight_RF, int y, int x) {
    PROFILE_FUNCTION();
    Vector3f transformed_F(3);
    Vector3f point(3);
    Vector3f solution(3);
    {
        PROFILE_SCOPE("RF Pre");
        // Transform F from 2D image space to 3D world space with C
        m23(F, Cx, Cy, transformed_F, y, x);
        // calculate cross product between world space F and calibration matrix.
        // this gives us the point on which the line stands
        crossProduct3x3(C, transformed_F, point, y, x);
        // right hand side B consists of a sum of a points
        // subtract the contribution of the old_point at y,x and add the contribution of the new point
        B = B - Identity_minus_outerProducts[y][x]*old_points[y][x] + Identity_minus_outerProducts[y][x]*point;
        // new point is now old
        old_points[y][x] = point;
    }
    // solve for the new rotation vector
    solution = A.partialPivLu().solve(B);
    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
}

void register_R_point(const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, int y, int x) {
    PROFILE_FUNCTION();
    // Replace this pixel's contribution to the F -> R right-hand side B by its current flow
    Vector3f transformed_F;
    Vector3f point;
    m23(F, Cx, Cy, transformed_F, y, x);
    crossProduct3x3(C, transformed_F, point, y, x);
    B += Identity_minus_outerProducts[y][x] * (point - old_points[y][x]);
    old_points[y][x] = point;
}

Vector2f m32_pixel(const Vector3f &v, const Tensor3f &Cx, const Tensor3f &Cy, int y, int x) {
    // Coefficients (channel 0 on Cy, channel 1 on Cx) of v in the tangent plane spanned by Cx and Cy. For v
    // in that plane (e.g. R x C, as |C| = 1) this equals m32 at the pixel.
    Matrix<float, 3, 2> M;
    M << Cy(y, x, 0), Cx(y, x, 0),
         Cy(y, x, 1), Cx(y, x, 1),
         Cy(y, x, 2), Cx(y, x, 2);
    // Normal equations, 2 x 2
    const Matrix2f N = M.transpose() * M;
    const Vector2f r = M.transpose() * v;
    const float det = N(0, 0) * N(1, 1) - N(0, 1) * N(1, 0);
    return Vector2f(N(1, 1) * r(0) - N(0, 1) * r(1), N(0, 0) * r(1) - N(1, 0) * r(0)) / det;
}

void relaxB(Vector3f &B, float &BTime, const Matrix3f &A, const Tensor1f &R, const float time, const float rate_FR) {
    PROFILE_FUNCTION();
    // Every pixel's F relaxes towards m32(R x C) at rate_FR, and a relaxed pixel contributes (I - C C^T) R to B,
    // so B relaxes towards A R at the same rate. Exact while R is constant, and R only changes at events.
    const float dt = time - BTime;
    BTime = time;
    if (dt <= 0.0f) return;
    const Vector3f AR = A * Vector3f(R(0), R(1), R(2));
    B = AR + (B - AR) * expf(-rate_FR * dt);
}

void relaxFR(Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, Tensor2f &FTimeSurface, std::vector<std::vector<Vector3f>> &old_points, const int y, const int x, const float time, const float rate_FR) {
    PROFILE_FUNCTION();
    // Lazy form of update_FR at one pixel: F relaxes towards m32(R x C) over the time since it was last brought
    // up to date, with R taken as its current value. B already contains this relaxation (relaxB), so only the
    // stored point is refreshed, not B.
    const float dt = time - FTimeSurface(y, x);
    FTimeSurface(y, x) = time;
    if (dt <= 0.0f) return;
    Vector3f RxC;
    const Vector3f c(C(y, x, 0), C(y, x, 1), C(y, x, 2));
    // (y, x, z) storage: cross product of the underlying (x, y, z) vectors, as in crossProduct1x3
    RxC(0) = R(2) * c(1) - R(1) * c(2);
    RxC(1) = R(0) * c(2) - R(2) * c(0);
    RxC(2) = R(1) * c(0) - R(0) * c(1);
    const Vector2f target = m32_pixel(RxC, Cx, Cy, y, x);
    const float keep = expf(-rate_FR * dt);
    F(y, x, 0) = target(0) + (F(y, x, 0) - target(0)) * keep;
    F(y, x, 1) = target(1) + (F(y, x, 1) - target(1)) * keep;
    Vector3f transformed_F;
    m23(F, Cx, Cy, transformed_F, y, x);
    crossProduct3x3(C, transformed_F, old_points[y][x], y, x);
}

void relaxFRNeighbourhood(Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, Tensor2f &FTimeSurface, std::vector<std::vector<Vector3f>> &old_points, const int y, const int x, const float time, const float rate_FR) {
    PROFILE_FUNCTION();
    // The event step and the silent G relaxation read F at (y, x) and its 4 neighbours.
    const int rows = static_cast<int>(F.dimension(0));
    const int cols = static_cast<int>(F.dimension(1));
    const int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto &offset : offsets) {
        const int yy = y + offset[0];
        const int xx = x + offset[1];
        if (yy < 0 || yy >= rows || xx < 0 || xx >= cols) continue;
        relaxFR(F, C, Cx, Cy, R, FTimeSurface, old_points, yy, xx, time, rate_FR);
    }
}

Tensor3f relaxedFRCopy(const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, const Tensor2f &FTimeSurface, const float time, const float rate_FR) {
    PROFILE_FUNCTION();
    Tensor3f relaxed = F;
    Tensor2f times = FTimeSurface;
    std::vector<std::vector<Vector3f>> points(F.dimension(0), std::vector<Vector3f>(F.dimension(1)));
    for (int y = 0; y < F.dimension(0); ++y) {
        for (int x = 0; x < F.dimension(1); ++x) {
            relaxFR(relaxed, C, Cx, Cy, R, times, points, y, x, time, rate_FR);
        }
    }
    return relaxed;
}

void update_RIMU(Tensor1f &R, const std::vector<float> &rotVelIMU, const float weight_RIMU, const Matrix3f &R_cam_imu, const float imuScale) {
    const Vector3f omega_cam = R_cam_imu * Vector3f(rotVelIMU[0], rotVelIMU[1], rotVelIMU[2]);   // (x, y, z)
    // R = -(camera rotation), stored as (y, x, z)
    const Vector3f candidate = -imuScale * Vector3f(omega_cam.y(), omega_cam.x(), omega_cam.z());
    R(0) = (1 - weight_RIMU)*R(0) + weight_RIMU*candidate(0);
    R(1) = (1 - weight_RIMU)*R(1) + weight_RIMU*candidate(1);
    R(2) = (1 - weight_RIMU)*R(2) + weight_RIMU*candidate(2);
}



