# How to Contribute to Tlama Core 🌟

Hola

Thank you for your interest in contributing to **Tlama Core**! Your contributions help improve the project and make it more useful for everyone. This project follows **Git Flow**, so please follow these steps when contributing.  

## 🛠️ **Git Flow Overview**  
We use the **Git Flow** branching model, which consists of:  
- **`main`** → Stable production-ready code.  
- **`develop`** → Main branch for active development.  
- **Feature branches (`feature/<name>`)** → New features are developed here.  
- **Release branches (`release/<version>`)** → Used for preparing new versions.  
- **Hotfix branches (`hotfix/<name>`)** → Critical fixes to `main`.  

### 🔀 **1️⃣ Fork and Clone the Repository**  
1. Fork the repository by clicking the **“Fork”** button on GitHub.  
2. Clone your forked repository:  
   ```sh
   git clone https://github.com/your-username/tlama-core.git
   cd tlama-core
   git checkout develop
   ```

### 🌱 **2️⃣ Create a Feature Branch**  
All new features should be developed in a feature branch:  
```sh
git checkout -b feature/<your-feature-name> develop
```

### ✏️ **3️⃣ Make Your Changes**  
- Follow the [Style Guide](style-guide.md) to ensure consistency.  
- If adding new features, update the documentation accordingly.  

### ✅ **4️⃣ Commit Your Changes**  
Write meaningful commit messages:  
```sh
git add .
git commit -m "feat: Added feature XYZ to improve ABC"
```

### 🚀 **5️⃣ Push to Your Fork**  
```sh
git push origin feature/<your-feature-name>
```

### 🔁 **6️⃣ Open a Pull Request (PR) to `develop`**  
- Go to the **original repository** on GitHub.  
- Click **“New Pull Request”**.  
- Select **your feature branch** and set the base branch to `develop`.  
- Follow the [Pull Request Template](pull-request-template.md).  

### 🧐 **7️⃣ Review Process**  
- Your PR will be reviewed by maintainers.  
- You may be asked for changes before approval.  
- Once approved, it will be merged into `develop`.  

---

## 🔥 **Fixing Bugs (Hotfixes)**  
For **critical bugs**, use a `hotfix/<name>` branch instead of `feature/<name>`:  
```sh
git checkout -b hotfix/<bug-name> main
# Apply fix
git commit -m "fix: Critical bug in XYZ"
git push origin hotfix/<bug-name>
```
- Open a PR to `main` and, once merged, also merge it into `develop`.  

---

## 🏷️ **Release Process**  
When a new version is ready, a `release/<version>` branch will be created:  
```sh
git checkout -b release/v1.0 develop
```
- Only bug fixes and documentation updates should be added here.  
- Once finalized, it will be merged into both `main` and `develop`.  

---

## 🎯 **Final Notes**  
- **Never push directly to `main` or `develop`**. Always use feature/hotfix branches.  
- Keep your fork updated with `develop`:  
  ```sh
  git fetch upstream
  git checkout develop
  git merge upstream/develop
  ```  
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for consistent commit messages.  

Thank you for contributing! 🚀  
