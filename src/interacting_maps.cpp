#include "interacting_maps.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

std::string create_folder_and_update_gitignore(const std::string& folder_name) {
    // Get the absolute path of the current working directory
    fs::path current_directory = fs::current_path();
    
    // Create the output folder if it does not exist
    fs::path output_folder_path = current_directory / "output";
    // Create the directory if it does not exist
    if (!fs::exists(output_folder_path)) {
        fs::create_directory(output_folder_path);
    }

    // Same for the actual folder
    fs::path folder_path = output_folder_path / folder_name;
    if (!fs::exists(folder_path)) {
        fs::create_directory(folder_path);
    }
    
    // Path to the .gitignore file
    fs::path gitignore_path = current_directory/ ".gitignore";
    
    // Check if the folder is already in .gitignore
    bool folder_in_gitignore = false;
    if (fs::exists(gitignore_path)) {
        std::ifstream gitignore_file(gitignore_path);
        std::string line;
        while (std::getline(gitignore_file, line)) {
            if (line == folder_name || line == "/" + folder_name) {
                folder_in_gitignore = true;
                break;
            }
        }
    }
    
    // Add the folder to .gitignore if it's not already there
    if (!folder_in_gitignore) {
        std::ofstream gitignore_file(gitignore_path, std::ios_base::app);
        gitignore_file << "\n" << folder_name << "\n";
    }
    
    // Return the absolute path of the new folder
    return folder_path.string();
}

int main() {
    std::string folder_name = "example_folder";
    std::string folder_path = create_folder_and_update_gitignore(folder_name);
    std::cout << "Folder created at: " << folder_path << std::endl;
    return 0;
}