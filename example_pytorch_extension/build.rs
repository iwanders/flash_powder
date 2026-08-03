// This is only here because otherwise 'cargo t' complains with a linker error, it's not actually necessary to build the
// extension.
fn main() {
    torch_stable::downtree_build_rs();
}
