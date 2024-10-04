#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
typedef Eigen::Triplet<double> T;

/******************This is the defined functions field like convolution, filter, export matrix, etc***********************/
// Definite the convolution function by matrix production, return the sparsematrix mn*mn
SparseMatrix<double, RowMajor> convolutionMatrix(const Matrix<double, Dynamic, Dynamic, RowMajor> &kernel, int height, int width)
{
    const int kernel_size = kernel.rows();
    const int m = height;
    const int n = width;
    const int mn = m * n;
    SparseMatrix<double, RowMajor> A(mn, mn);

    std::vector<T> hav2TripletList;
    hav2TripletList.reserve(mn * kernel_size * kernel_size);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int index_Ai = i * n + j; // row number of A
            // ki, kj are respectively row index and column index of kernel, central point is (0,0)
            for (int ki = -kernel_size / 2; ki <= kernel_size / 2; ki++)
            {
                for (int kj = -kernel_size / 2; kj <= kernel_size / 2; kj++)
                {
                    int ci = i + ki; // contribute to n(width) shift each time when there is a vertical moving for convolution or inside the kernel
                    int cj = j + kj; // contribute just 1 shift each when there is a horizontal moving for convilution or inside the kernel
                    if (ci >= 0 && ci < m && cj >= 0 && cj < n && kernel(ki + kernel_size / 2, kj + kernel_size / 2) != 0)
                    {
                        int index_Aj = ci * n + cj; // column number of A
                        hav2TripletList.push_back(Triplet<double>(index_Ai, index_Aj, kernel(ki + kernel_size / 2, kj + kernel_size / 2)));
                    }
                }
            }
        }
    }
    // get the sparsematrix from tripletlist
    A.setFromTriplets(hav2TripletList.begin(), hav2TripletList.end());
    return A;
}

// An alternative definition for kernels H 3 by 3, easy to generalize
SparseMatrix<double, RowMajor> convolutionMatrix2(const Matrix<double, Dynamic, Dynamic, RowMajor> &kernel, int m, int n)
{
    const int mn = m * n;
    SparseMatrix<double, RowMajor> A(mn, mn);
    std::vector<T> tripletList;
    tripletList.reserve(mn * 9);
    for (int i = 0; i < mn; ++i)
    {
        // top center (not first n rows)
        if (i - n + 1 > 0 && kernel(0, 1) != 0)
            tripletList.push_back(T(i, i - n, kernel(0, 1)));
        // middle center (always)
        if (kernel(1, 1) != 0)
            tripletList.push_back(T(i, i, kernel(1, 1)));
        // bottom center (not last n rows)
        if (i + n - 1 < mn - 1 && kernel(2, 1) != 0)
            tripletList.push_back(T(i, i + n, kernel(2, 1)));

        if (i % n != 0) // we can go left
        {
            // top left
            if (i - n > 0 && kernel(0, 0) != 0)
                tripletList.push_back(T(i, i - n - 1, kernel(0, 0)));
            // middle left
            if (i > 0 && kernel(1, 0) != 0)
                tripletList.push_back(T(i, i - 1, kernel(1, 0)));
            // bottom left
            if (i + n - 2 < mn - 1 && kernel(2, 0) != 0)
                tripletList.push_back(T(i, i + n - 1, kernel(2, 0)));
        }

        if ((i + 1) % n != 0) // we can go right
        {
            // top right
            if (i - n + 2 > 0 && kernel(0, 2) != 0)
                tripletList.push_back(T(i, i - n + 1, kernel(0, 2)));
            // middle right
            if (i < mn - 1 && kernel(1, 2) != 0)
                tripletList.push_back(T(i, i + 1, kernel(1, 2)));
            // bottom right
            if (i + n < mn - 1 && kernel(2, 2) != 0)
                tripletList.push_back(T(i, i + n + 1, kernel(2, 2)));
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
}

// Define the function that convert a vector to a Matrix<unsigned char> type and output it to image.png
void outputVectorImage(const VectorXd &vectorData, int height, int width, const std::string &path)
{
    Matrix<double, Dynamic, Dynamic, RowMajor> output_image_matrix(height, width);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            output_image_matrix(i, j) = vectorData(i * width + j);
        }
    }

    // Convert the modified image to grayscale and export it using stbi_write_png
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> new_image_output = output_image_matrix.unaryExpr(
        [](double pixel)
        {
            return static_cast<unsigned char>(std::max(0.0, std::min(255.0, pixel * 255))); // ensure range [0,255]
        });
    if (stbi_write_png(path.c_str(), width, height, 1, new_image_output.data(), width) == 0)
    {
        std::cerr << "Error: Could not save modified image" << std::endl;
    }
    std::cout << "New image saved to " << path << "\n"
              << std::endl;
}

// Export the vector
void exportVector(VectorXd data, const std::string &path)
{
    FILE *out = fopen(path.c_str(), "w");
    fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out, "%d\n", data.size());
    for (int i = 0; i < data.size(); i++)
    {
        fprintf(out, "%d %f\n", i + 1, data(i)); // here index from 1
    }
    fclose(out);
}

// Export a sparse matrix by saveMarket
void exportSparsematrix(SparseMatrix<double, RowMajor> data, const std::string &path)
{
    if (saveMarket(data, path))
    {
        std::cout << "Sparse matrix A saved to " << path << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not save sparse matrix A to " << path << std::endl;
    }
}

// Check if a matrix is symmetric by tolerance 1e-10
bool isSymmetric(SparseMatrix<double, RowMajor> &matrix, const std::string &matrixName)
{
    double tolerance = 1e-10;
    double norm_diff = (matrix - SparseMatrix<double, RowMajor>(matrix.transpose())).norm();
    std::cout << matrixName << "\trow: " << matrix.rows() << "\tcolumns: " << matrix.cols() << std::endl;
    std::cout << "Check if " << matrixName << " is symmectric by norm value of its difference with transpose: "
              << norm_diff << std::endl;
    return norm_diff < tolerance;
}

// Function to check if a matrix is positive definite by cholesky
bool isPositiveDefinite(const SparseMatrix<double, RowMajor> &matrix)
{
    Eigen::SimplicialLLT<SparseMatrix<double, RowMajor>> cholesky(matrix);
    return cholesky.info() == Success;
}

// 1. Lis generated mtx file is marketvector format, but loadMarkerVector() method doesn't match (it needs MatrixMarket matrix array fromat);
// 2. So we use our own menthod to read data from mtx file here, we read each line of the file and put it value into our Eigen::VectorXd data.
// Function to read a vector from a Matrix Market file with 0-based indexing
VectorXd readMarketVector(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return VectorXd();
    }

    std::string line;
    // Skip the header lines
    std::getline(file, line); // %%MatrixMarket vector coordinate real general
    std::getline(file, line); // Dimensions or size of the vector

    int size;
    std::istringstream iss(line);
    if (!(iss >> size))
    {
        std::cerr << "Error reading size from file: " << filename << std::endl;
        return VectorXd();
    }

    VectorXd vectorX(size);

    // Read the vector data line by line, data index staring from 1
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        int index;
        double value;
        if (!(iss >> index >> value))
        {
            std::cerr << "Error reading value from file: " << filename << std::endl;
            return VectorXd();
        }
        // Convert 1-based to 0-based indexing here, because our VectorXd type stores data from 0
        vectorX(index - 1) = value;
    }

    file.close();
    return vectorX;
}

/*-------------------------------------------------Main()-------------------------------------------------------*/
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char *input_image_path = argv[1];

    /*****************************Load the image using stb_image****************************/
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data)
    {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image " << argv[1] << " loaded: " << height << "x" << width << " pixels with " << channels << " channels" << std::endl;

    /****************Convert the image_data to MatrixXd form, each element value [0,1]*******************/
    // Attention! if use MatrixXd here its columnmajor by default, we prefere rowmajor
    Matrix<double, Dynamic, Dynamic, RowMajor> original_image_matrix(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * channels;
            original_image_matrix(i, j) = static_cast<double>(image_data[index]) / 255;
        }
    }

    // Report the size of the matrix
    std::cout << "The original image " << argv[1] << " in matrix form has dimension: " << original_image_matrix.rows() << " rows x " << original_image_matrix.cols()
              << " cols = " << original_image_matrix.size() << "\n"
              << std::endl;

    /***********************************Introduce noise and export****************************************/
    Matrix<double, Dynamic, Dynamic, RowMajor> noised_image_matrix(height, width);
    // Uniform random number generator for noise
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(-50, 50);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int noise = distribution(generator);
            double noisedData = original_image_matrix(i, j) + static_cast<double>(noise) / 255;
            noised_image_matrix(i, j) = std::max(0.0, std::min(1.0, noisedData)); // ensure the value in [0,1]
        }
    }

    // Convert noise image to grayscale and export it by using stbi_write_png
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noised_image_output = noised_image_matrix.unaryExpr(
        [](double pixel)
        { return static_cast<unsigned char>(pixel * 255); });
    const std::string noised_image_path = "output_NoisedImage.png";
    if (stbi_write_png(noised_image_path.c_str(), width, height, 1,
                       noised_image_output.data(), width) == 0)
    {
        std::cerr << "Error: Could not save noised image" << std::endl;
        return 1;
    }
    std::cout << "New image saved to " << noised_image_path << "\n"
              << std::endl;
    /*************************************************end****************************************************/

    /*-------------------By map creat a vector reference to memeory without copying data----------------------*/
    // It is columnmajor by default, however, we've already declared our data rowmajor so here rowmajor as well.
    Map<VectorXd> v(original_image_matrix.data(), original_image_matrix.size());
    Map<VectorXd> w(noised_image_matrix.data(), noised_image_matrix.size());

    // Verify the size of the vectors
    std::cout << "Original image vector v's size: " << v.size() << std::endl;
    std::cout << "Noisy image vector w's size: " << w.size() << std::endl;
    std::cout << "Euclidean norm of v is: " << v.norm() << std::endl;

    /****************************Create different kernels and perform convolution*****************************/
    const int kernel_size = 3;

    // Create the kernel H_{av2}
    const double hav2_value = 1.0 / (kernel_size * kernel_size);
    Matrix<double, kernel_size, kernel_size, RowMajor> hav2;
    hav2.setConstant(hav2_value);
    // Perform convolution function
    SparseMatrix<double, RowMajor> A1 = convolutionMatrix2(hav2, height, width);
    // Check the nonzero numbers
    std::cout << "A1 nonzero number is: " << A1.nonZeros() << std::endl;
    // Smooth the noise image by using this filterImage function and passing data and path into it
    const std::string smooth_image_path = "output_SmoothedImage.png";
    VectorXd smoothed_image_vector = A1 * w;
    outputVectorImage(smoothed_image_vector, height, width, smooth_image_path);

    // Create the kernel H_{sh2}}
    Matrix<double, kernel_size, kernel_size, RowMajor> hsh2;
    hsh2 << 0.0, -3.0, 0.0,
        -1.0, 9.0, -3.0,
        0.0, -1.0, 0.0;
    // Perform convolution function
    SparseMatrix<double, RowMajor> A2 = convolutionMatrix2(hsh2, height, width);
    // Check the nonzero numbers
    std::cout << "A2 nonzero number is: " << A2.nonZeros() << std::endl;
    // Verify if the matrix A2 is symmetric
    isSymmetric(A2, "A2") ? std::cout << "The matrix A2 is symmetric!" << std::endl
                          : std::cout << "The matrix A2 is not symmetric!" << std::endl;

    // Sharpen the original image by matrix production
    const std::string sharpen_image_path = "output_SharpenedImage.png";
    VectorXd sharpened_image_vector = A2 * v;
    outputVectorImage(sharpened_image_vector, height, width, sharpen_image_path);

    // Create the kernel H_{sh2}}
    Matrix<double, kernel_size, kernel_size, RowMajor> hlap;
    hlap << 0.0, -1.0, 0.0,
        -1.0, 4.0, -1.0,
        0.0, -1.0, 0.0;
    // Perform convolution function
    SparseMatrix<double, RowMajor> A3 = convolutionMatrix2(hlap, height, width);
    // Verify if the matrix A3 is symmetric
    isSymmetric(A3, "A3") ? std::cout << "The matrix A3 is symmetric!" << std::endl
                          : std::cout << "The matrix A3 is not symmetric!" << std::endl;

    // Check if A3 is positive definite as well
    isPositiveDefinite(A3) ? std::cout << "The matrix A3 is positive definite!" << std::endl
                           : std::cout << "The matrix A3 is not positive definite!" << std::endl;

    // Edge detection of the original image
    const std::string edgeDetection_image_path = "output_EdgeDetectionImage.png";
    VectorXd edgeDetected_image_vector = A3 * v;
    outputVectorImage(edgeDetected_image_vector, height, width, edgeDetection_image_path);
    /************************************************end*******************************************************/

    /**********************************Solve equation of (I + A3)*y = w****************************************/
    VectorXd y(w.size());
    SparseMatrix<double, RowMajor> I(A3.rows(), A3.rows());
    I.setIdentity();
    SparseMatrix<double, RowMajor> A3_Plus_I = A3 + I;
    ConjugateGradient<SparseMatrix<double, RowMajor>, Lower | Upper> cg; // A3 is spd
    cg.setTolerance(1e-10);
    cg.compute(A3_Plus_I);
    y = cg.solve(w);
    std::cout << "The iteration count is: " << cg.iterations() << std::endl;
    std::cout << "The final residual is: " << cg.error() << std::endl;
    // Output the y vector to image
    const std::string y_image_path = "output_VectorY.png";
    outputVectorImage(y, height, width, y_image_path);
    /************************************************end*******************************************************/

    // Export the sparse matrix A1 A2 A3
    const std::string sparse_matrixA1_path = "./A1.mtx";
    exportSparsematrix(A1, sparse_matrixA1_path);
    const std::string sparse_matrixA2_path = "./A2.mtx";
    exportSparsematrix(A2, sparse_matrixA2_path);
    const std::string sparse_matrixA3_path = "./A3.mtx";
    exportSparsematrix(A3, sparse_matrixA3_path);

    // Export vector v, w and y
    const std::string vpath = "./v.mtx";
    exportVector(v, vpath);
    const std::string wpath = "./w.mtx";
    exportVector(w, wpath);
    const std::string ypath = "./y.mtx";
    exportVector(y, ypath);

    /**********************************Solve equation of A2*x = w****************************************/
    const std::string xMtxPath = "x.mtx";
    VectorXd x = readMarketVector(xMtxPath); // Read x.mtx file data and output it as image
    const std::string x_image_path = "output_VectorX.png";
    outputVectorImage(x, height, width, x_image_path);
    /*********************************************end****************************************************/

    // Free memory
    stbi_image_free(image_data);

    return 0;
}