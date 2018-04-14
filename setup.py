from setuptools import setup


requirements = []
dep_links    = []

with open('requirements.txt') as reqs:
    for r in reqs:
        if "git+https" in r:
            egg = r.split("#egg=")[1].strip()
            requirements.append(egg.replaces("-", "=="))
            link = r.replace("git+https", "http").replace(".git@", "/tarball")
            dep_links.append(link)
        else:
            requirements.append(r.strip())

setup(
      name = 'api_z',
      packages = ['api_z'],
      install_requirements = requirements,
      dependency_links = dep_links,
    )
