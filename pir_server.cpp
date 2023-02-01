
//
// Created by Haris Mughees on 4/21/21.
//

#include "pir_server.h"


pir_server::pir_server(const EncryptionParameters &params, const PirParams &pir_params) :
        params_(params),
        pir_params_(pir_params),
        is_db_preprocessed_(false)
{
    auto context = SEALContext::Create(params, false);
    evaluator_ = make_unique<Evaluator>(context);
    seal::GaloisKeys galkey;
    parms_id_= context->first_parms_id();
    newcontext_ = SEALContext::Create(params_);

}

void pir_server::set_galois_key(std::uint32_t client_id, seal::GaloisKeys galkey) {
    //galkey.parms_id() = parms_id_;
    galoisKeys_ = galkey;

}

std::vector<uint64_t *> pir_server::get_split_db_at(uint64_t index) {
    int decomp_size = params_.plain_modulus().bit_count() / pir_params_.plain_base;

    const auto &context_data2 = newcontext_->first_context_data();
    auto &parms2 = context_data2->parms();
    auto &coeff_modulus = parms2.coeff_modulus();
    size_t coeff_mod_count = coeff_modulus.size();
    size_t coeff_count = parms2.poly_modulus_degree();

    std::vector<uint64_t*> indexed;
    for (int i = 0; i < decomp_size; i++) {
        indexed.push_back(split_db_data + (((index * decomp_size) + i) * coeff_count * coeff_mod_count));
    }
    return indexed;
}

void pir_server::set_database_from_file(std::uint64_t ele_num, std::uint64_t ele_size) {
    u_int64_t logt = params_.plain_modulus().bit_count();
    uint32_t N = params_.poly_modulus_degree();
    // number of FV plaintexts needed to represent all elements
    uint64_t total = plaintexts_per_db(logt, N, ele_num, ele_size);
    int decomp_size = params_.plain_modulus().bit_count() / pir_params_.plain_base;

    const auto &context_data2 = newcontext_->first_context_data();
    auto &parms2 = context_data2->parms();
    auto &coeff_modulus = parms2.coeff_modulus();
    size_t coeff_mod_count = coeff_modulus.size();
    size_t coeff_count = parms2.poly_modulus_degree();

     // number of FV plaintexts needed to create the d-dimensional matrix
    uint64_t prod = 1;
    for (uint32_t i = 0; i < pir_params_.nvec.size(); i++) {
        prod *= pir_params_.nvec[i];
    }

    uint64_t matrix_plaintexts = prod;
    assert(total <= matrix_plaintexts);


    size_t split_db_data_elements = matrix_plaintexts * decomp_size * coeff_count * coeff_mod_count;
    split_db_data_size = split_db_data_elements * sizeof(uint64_t);

    int split_db_fd = open("split_db.db", O_RDONLY | O_LARGEFILE, S_IRWXU);
    if (split_db_fd == -1) {
        printf("File open failed with error: %s\n", strerror(errno));
        throw std::runtime_error("File open failed");
    }

    split_db_data = (uint64_t*) mmap(NULL, split_db_data_size, PROT_READ, MAP_SHARED, split_db_fd, 0);
    if (split_db_data == MAP_FAILED) {
        printf("Tried to allocate %ld bytes\n", split_db_data_size);
        printf("Mapping failed with error %s\n", strerror(errno));
        throw std::runtime_error("Mapping failed");
    }
}

void
pir_server::set_database(const unique_ptr<const std::uint8_t[], MmapDeleter> &bytes, std::uint64_t ele_num, std::uint64_t ele_size) {
    u_int64_t logt = params_.plain_modulus().bit_count();
    uint32_t N = params_.poly_modulus_degree();
    // number of FV plaintexts needed to represent all elements
    uint64_t total = plaintexts_per_db(logt, N, ele_num, ele_size);
    int decomp_size = params_.plain_modulus().bit_count() / pir_params_.plain_base;

    const auto &context_data2 = newcontext_->first_context_data();
    auto &parms2 = context_data2->parms();
    auto &coeff_modulus = parms2.coeff_modulus();
    size_t coeff_mod_count = coeff_modulus.size();
    size_t coeff_count = parms2.poly_modulus_degree();



    // number of FV plaintexts needed to create the d-dimensional matrix
    uint64_t prod = 1;
    for (uint32_t i = 0; i < pir_params_.nvec.size(); i++) {
        prod *= pir_params_.nvec[i];
    }



    uint64_t matrix_plaintexts = prod;
    assert(total <= matrix_plaintexts);

    uint64_t ele_per_ptxt = elements_per_ptxt(logt, N, ele_size);


    uint64_t bytes_per_ptxt = ele_per_ptxt * ele_size;

    uint64_t db_size = ele_num * ele_size;

    uint64_t coeff_per_ptxt = ele_per_ptxt * coefficients_per_element(logt, ele_size);
    cout<<"total==>"<< logt <<endl;
    assert(coeff_per_ptxt <= N);

    cout << "Server: total number of FV plaintext = " << total << endl;
    cout << "Server: elements packed into each plaintext " << ele_per_ptxt << endl;

    uint64_t offset = 0;

    uint64_t current_plaintexts = 0;

    size_t split_db_data_elements = matrix_plaintexts * decomp_size * coeff_count * coeff_mod_count;
    split_db_data_size = split_db_data_elements * sizeof(uint64_t);

    ofstream split_db_file;
    split_db_file.open("split_db.db", ios::binary | ios::trunc);

    uint64_t* decomp_space = (uint64_t*) malloc(coeff_count * coeff_mod_count * sizeof(uint64_t));

    for (uint64_t i = 0; i < total; i++) {

        uint64_t process_bytes = 0;


        if (db_size <= offset) {
            break;
        } else if (db_size < offset + bytes_per_ptxt) {
            process_bytes = db_size - offset;
        } else {
            process_bytes = bytes_per_ptxt;
        }

        // Get the coefficients of the elements that will be packed in plaintext i
//        bytes_to_coeffs(std::uint32_t limit, const std::uint64_t *bytes,
//        std::uint64_t size);




        vector<uint64_t> coefficients = bytes_to_coeffs(logt, bytes.get() + offset, process_bytes);
            offset += process_bytes;

        uint64_t used = coefficients.size();

        assert(used <= coeff_per_ptxt);

        // Pad the rest with 1s
        for (uint64_t j = 0; j < (N - used); j++) {
            coefficients.push_back(1);
        }
//
        Plaintext plain;
        vector_to_plaintext(coefficients, plain);
        //cout << i << "-th encoded plaintext = " << plain.to_string() << endl;
        std::vector<uint64_t *> plain_decom;
        plain_decompositions(plain, newcontext_, decomp_size, pir_params_.plain_base, plain_decom);
        poc_nfllib_ntt_rlwe_decomp(plain_decom);

        for (int d = 0; d < decomp_size; d++) {
            split_db_file.write((char*)plain_decom[d], coeff_count * coeff_mod_count * sizeof(uint64_t));
            free(plain_decom[d]);
        }
        // split_db_file.flush();

        // split_db.push_back(plain_decom);
        current_plaintexts++;
    }

    // Add padding to make database a matrix
    assert(current_plaintexts <= total);
//
#ifdef DEBUG
    cout << "adding: " << matrix_plaintexts - current_plaintexts
         << " FV plaintexts of padding (equivalent to: "
         << (matrix_plaintexts - current_plaintexts) * elements_per_ptxt(logtp, N, ele_size)
         << " elements)" << endl;
#endif

    vector<uint64_t> padding(N, 1);

    for (uint64_t i = 0; i < (matrix_plaintexts - current_plaintexts); i++) {
        Plaintext plain;
        vector_to_plaintext(padding, plain);
        std::vector<uint64_t *> plain_decom;
        plain_decompositions(plain, newcontext_, decomp_size, pir_params_.plain_base, plain_decom, decomp_space);
        poc_nfllib_ntt_rlwe_decomp(plain_decom);
        for (int d = 0; d < decomp_size; d++) {
            split_db_file.write((char*)plain_decom[d], coeff_count * coeff_mod_count * sizeof(uint64_t));
            // free(plain_decom[d]);
        }
        // split_db_file.flush();
        // split_db.push_back(plain_decom);
    }

    split_db_file.close();

    int split_db_fd = open("split_db.db", O_RDONLY | O_LARGEFILE, S_IRWXU);
    if (split_db_fd == -1) {
        printf("File open failed with error: %s\n", strerror(errno));
        throw std::runtime_error("File open failed");
    }

    split_db_data = (uint64_t*) mmap(NULL, split_db_data_size, PROT_READ, MAP_SHARED, split_db_fd, 0);
    if (split_db_data == MAP_FAILED) {
        printf("Tried to allocate %ld bytes\n", split_db_data_size);
        printf("Mapping failed with error %s\n", strerror(errno));
        throw std::runtime_error("Mapping failed");
    }

    free(decomp_space);

    // // Preload pages from disk
    // std::uint64_t temp;
    // for (int i = 0; i < split_db_data_elements; i++) {
    //     temp += split_db_data[i];
    // }
    // printf("Sum: %ld\n", temp);
    // printf("Data size: %ld\n", split_db_data_size);
}

PirReply pir_server::generate_reply(PirQuery query, uint32_t client_id, SecretKey sk) {
    assert(query.size()==2);

    Decryptor dec(newcontext_,sk);
    Plaintext pt;
    pt.resize(4096);
    pt.set_zero();
    //pt[0]=123;

    vector<uint64_t> nvec = pir_params_.nvec;


    uint64_t product = 1;

    for (uint32_t i = 0; i < nvec.size(); i++) {
        product *= nvec[i];

    }

    auto coeff_count = params_.poly_modulus_degree();


    vector<Plaintext> intermediate_plain; // decompose....

    auto pool = MemoryManager::GetPool();


    int N = params_.poly_modulus_degree();

    int logt = params_.plain_modulus().bit_count();

    vector<Ciphertext> first_dim_intermediate_cts(product/nvec[0]);


    for (uint32_t i = 0; i < 1; i++) {



        uint64_t n_i = nvec[i];
        cout << "Server: first dim size = " << n_i << endl;
        cout << "Server: expanding " << query[i].size() << " query ctxts" << endl;

            uint64_t total = n_i;

            cout << "-- expanding one query ctxt into " << total  << " ctxts "<< endl;

            vector<GSWCiphertext> list_enc;

        int decomp_size = params_.plain_modulus().bit_count() / pir_params_.plain_base;
            list_enc.resize(n_i, GSWCiphertext(decomp_size));
            vector<GSWCiphertext>::iterator list_enc_ptr = list_enc.begin();
        auto start = high_resolution_clock::now();

            //vector<Ciphertext> expanded_query_part = expand_query(query[i][j], total, client_id);

            //n_1=64
            poc_expand_flat(list_enc_ptr, query[i], newcontext_, n_i, galoisKeys_);




        //cout<<"tttttt=========="<<time_server_us/query[i].size()<<endl;
        //cout << "Server: expansion done " << endl;
        // cout << " size mismatch!!! " << expanded_query.size() << ", " << n_i << endl;
        if (list_enc.size() != n_i) {
            cout << " size mismatch!!! " << list_enc.size() << ", " << n_i << endl;
        }


        for (uint32_t jj = 0; jj < list_enc.size(); jj++)
        {


            poc_nfllib_ntt_gsw(list_enc[jj],newcontext_);


        }

        auto end = high_resolution_clock::now();
        int time_server_us =  duration_cast<milliseconds>(end - start).count();
        cout<<"Rlwe exansion time= "<<time_server_us<<" ms"<<endl;


        product /= n_i;

        vector<Ciphertext> intermediateCtxts(product);


        auto time_server_s = high_resolution_clock::now();


        int durrr =0;
        for (uint64_t k = 0; k < product; k++) {

            first_dim_intermediate_cts[k].resize(newcontext_, newcontext_->first_context_data()->parms_id(), 2);
            std::vector<uint64_t *> split_db_k = get_split_db_at(k);
            poc_nfllib_external_product(list_enc[0], split_db_k, newcontext_, decomp_size, first_dim_intermediate_cts[k],1);

            for (uint64_t j = 1; j < n_i; j++) {

                uint64_t total = n_i;

                //cout << "-- expanding one query ctxt into " << total  << " ctxts "<< endl;


                Ciphertext temp;
                temp.resize(newcontext_, newcontext_->first_context_data()->parms_id(), 2);

                auto expand_start = high_resolution_clock::now();
                std::vector<uint64_t *> split_db_k_j = get_split_db_at(k + j * product);
                poc_nfllib_external_product(list_enc[j], split_db_k_j, newcontext_, decomp_size, temp,1);
                auto expand_end  = high_resolution_clock::now();

                evaluator_->add_inplace(first_dim_intermediate_cts[k], temp); // Adds to first component.
                //poc_nfllib_add_ct(first_dim_intermediate_cts[k], temp,newcontext_);


                 durrr = durrr+  duration_cast<microseconds>(expand_end - expand_start).count();
                //cout << "first-dimension cost" << durrr  << endl;


            }

        }


        cout << "first-dimension cost" << durrr/(product*n_i)  << endl;



       auto expand_start  = high_resolution_clock::now();

        for (uint32_t jj = 0; jj < first_dim_intermediate_cts.size(); jj++) {


            //evaluator_->transform_from_ntt_inplace(intermediateCtxts[jj]);
            poc_nfllib_intt_ct(first_dim_intermediate_cts[jj],newcontext_);

        }

        auto expand_end  = high_resolution_clock::now();
         durrr =  duration_cast<milliseconds>(expand_end - expand_start).count();
        cout << "INTT after first dimension" << durrr  << endl;


    }


    uint64_t  new_dimension_size=0, logsize;
    if(nvec.size()>1) {

        for (uint32_t i = 1; i < nvec.size(); i++) {
            new_dimension_size = new_dimension_size + nvec[i];
        }

        logsize = ceil(log2(new_dimension_size));

    }

    //testing starts from here
    vector<GSWCiphertext> CtMuxBits;
    int size = (1 << logsize);


    //int decomp_size = newcontext_->first_context_data()->total_coeff_modulus_bit_count() / pir_params_.gsw_base;
    int decomp_size = pir_params_.gsw_decomp_size ;
    int sk_decomp_size = newcontext_->first_context_data()->total_coeff_modulus_bit_count() / pir_params_.secret_base;
    CtMuxBits.resize((1 << logsize), GSWCiphertext(2 * decomp_size));
    vector<GSWCiphertext>::iterator gswCiphers_ptr = CtMuxBits.begin();


    thread_server_expand(gswCiphers_ptr, query[1], newcontext_, 0, decomp_size, size, galoisKeys_,  decomp_size, pir_params_.gsw_base, sk_decomp_size, pir_params_.secret_base, sk_enc_);

    for (uint32_t jj = 0; jj < CtMuxBits.size(); jj++)
    {
        poc_nfllib_ntt_gsw(CtMuxBits[jj],newcontext_);
    }

    auto expand_start = std::chrono::high_resolution_clock::now();
    //for remaining dimensions we treat them differently
    uint64_t  previous_dim=0;
    for (uint32_t i = 1; i < nvec.size(); i++){

        uint64_t n_i = nvec[i];

        product /= n_i;
        vector<Ciphertext> intermediateCtxts(product);//output size of this dimension

        for (uint64_t k = 0; k < product; k++) {


            intermediateCtxts[k].resize(newcontext_, newcontext_->first_context_data()->parms_id(), 2);
            std::vector<uint64_t *> rlwe_decom;
            rwle_decompositions(first_dim_intermediate_cts[k], newcontext_, decomp_size, pir_params_.gsw_base, rlwe_decom);
            poc_nfllib_ntt_rlwe_decomp(rlwe_decom);
            poc_nfllib_external_product(CtMuxBits[0 + previous_dim], rlwe_decom, newcontext_, decomp_size, intermediateCtxts[k],1);
            for (auto p : rlwe_decom) {
                free(p);
            }

            for (uint64_t j = 1; j < n_i; j++) {


                Ciphertext temp;
                rlwe_decom.clear();
                rwle_decompositions(first_dim_intermediate_cts[k + j * product], newcontext_, decomp_size, pir_params_.gsw_base, rlwe_decom);
                poc_nfllib_ntt_rlwe_decomp(rlwe_decom);
                temp.resize(newcontext_, newcontext_->first_context_data()->parms_id(), 2);
                poc_nfllib_external_product(CtMuxBits[j + previous_dim], rlwe_decom, newcontext_, decomp_size, temp,1);

                for (auto p : rlwe_decom) {
                    free(p);
                }
                evaluator_->add_inplace(intermediateCtxts[k], temp); // Adds to first component.



            }

        }



        for (uint32_t jj = 0; jj < intermediateCtxts.size(); jj++) {

            poc_nfllib_intt_ct(intermediateCtxts[jj],newcontext_);

        }

        first_dim_intermediate_cts.clear();
        first_dim_intermediate_cts=intermediateCtxts;
        previous_dim=previous_dim+n_i;

    }


    auto expand_end  = high_resolution_clock::now();
    int durrr =  duration_cast<milliseconds>(expand_end - expand_start).count();
    cout << "remaining-dimensions cost" << durrr  << endl;


    auto Total_end  = high_resolution_clock::now();
    durrr =  duration_cast<milliseconds>(Total_end - expand_start).count();
    cout << "remaining-dimensions cost" << durrr  << endl;

    return first_dim_intermediate_cts;
}



PirReply pir_server::generate_reply_combined(PirQuery query, uint32_t client_id, SecretKey sk) {
    // DiskTracker disk_tracker(100);
    // disk_tracker.start();

    assert(query.size()==2);

    auto Total_start  = high_resolution_clock::now();

    Decryptor dec(newcontext_,sk);
    Plaintext pt;
    pt.resize(4096);
    pt.set_zero();
    //pt[0]=123;

    vector<uint64_t> nvec = pir_params_.nvec;


    uint64_t product = 1;

    for (uint32_t i = 0; i < nvec.size(); i++) {
        product *= nvec[i];

    }


    vector<Plaintext> intermediate_plain; // decompose....

    auto pool = MemoryManager::GetPool();


    int N = params_.poly_modulus_degree();
    
    const auto &context_data2 = newcontext_->first_context_data();
    auto &parms2 = context_data2->parms();
    auto &coeff_modulus = parms2.coeff_modulus();
    size_t coeff_mod_count = coeff_modulus.size();
    size_t coeff_count = parms2.poly_modulus_degree();

    int logt = params_.plain_modulus().bit_count();

    vector<Ciphertext> first_dim_intermediate_cts(product/nvec[0]);


    for (uint32_t i = 0; i < 1; i++) {

        uint64_t n_i = nvec[i];
        cout << "Server: first dimension size  = " << n_i << endl;


        uint64_t total = n_i;



        vector<GSWCiphertext> list_enc;

        int decomp_size = params_.plain_modulus().bit_count() / pir_params_.plain_base;
        list_enc.resize(n_i, GSWCiphertext(decomp_size));
        vector<GSWCiphertext>::iterator list_enc_ptr = list_enc.begin();
        auto start = high_resolution_clock::now();

        //vector<Ciphertext> expanded_query_part = expand_query(query[i][j], total, client_id);

        //n_1=64
        poc_expand_flat_combined(list_enc_ptr, query[i], newcontext_, n_i, galoisKeys_);





        //cout << "Server: expansion done " << endl;
        // cout << " size mismatch!!! " << expanded_query.size() << ", " << n_i << endl;
        if (list_enc.size() != n_i) {
            cout << " size mismatch!!! " << list_enc.size() << ", " << n_i << endl;
        }


        for (uint32_t jj = 0; jj < list_enc.size(); jj++)
        {


            poc_nfllib_ntt_gsw(list_enc[jj],newcontext_);


        }

        auto end = high_resolution_clock::now();
        int time_server_us =  duration_cast<milliseconds>(end - start).count();
        cout<<"Server: rlwe exansion time = "<<time_server_us<<" ms"<<endl;


        product /= n_i;

        uint64_t num_coeff = coeff_count * coeff_mod_count;

        int durrr =0;

        auto expand_start = high_resolution_clock::now();

        // 1 x n_i matrix of GSW ciphertexts (2 x 2l matrices)
        // Will be multiplied with n_i x product matrix (database) of RLWE plaintexts (2l x 1 matrices)
        uint64_t* queryMatrix = (uint64_t*)malloc(n_i * decomp_size * 2 * num_coeff * sizeof(uint64_t));
        // Result 1 x product matrix of ciphertexts (2 x 1 matrices)
        uint64_t* intermediateCiphertexts = (uint64_t*)malloc(product * 2 * num_coeff * sizeof(uint64_t));

        // Copy ciphertext data into matrix pointer
        for (uint64_t k = 0; k < n_i; k++) {
            for (uint64_t d = 0; d < decomp_size; d++) {
                for (uint64_t c = 0; c < 2; c++) {
                    memcpy((void*)&queryMatrix[k * decomp_size * 2 * num_coeff + c * decomp_size * num_coeff + d * num_coeff], 
                                        (void*)list_enc[k][d].data(c), num_coeff * sizeof(uint64_t));
                }
            }
        }

        // Copy coefficient moduli information into vector
        std::vector<uint64_t> coeff_moduli;
        std::vector<uint128_t> m;

        for (int i = 0; i < coeff_modulus.size(); i++) {
            coeff_moduli.push_back(coeff_modulus[i].value());
            const uint64_t* const_ratio_ = coeff_modulus[i].const_ratio().data();
            uint128_t const_ratio = ((uint128_t)(const_ratio_[1]) << 64) + (uint128_t)(const_ratio_[0]);
            m.push_back(const_ratio);
        }

        // Do matrix multiplication
        mul_matrix_matrix_mod(queryMatrix, 1, n_i, 2, decomp_size, split_db_data, n_i, product, decomp_size, 1, 
                                intermediateCiphertexts, coeff_count, coeff_moduli, m);


        // Copy results into ciphertext objects
        for (uint64_t k = 0; k < product; k++) {
            first_dim_intermediate_cts[k].resize(newcontext_, newcontext_->first_context_data()->parms_id(), 2);
            for (uint64_t c = 0; c < 2; c++) {
                poly_nfllib_add(&intermediateCiphertexts[k * 2 * num_coeff + c * num_coeff], first_dim_intermediate_cts[k].data(c), first_dim_intermediate_cts[k].data(c));
            }
        }

        auto expand_end  = high_resolution_clock::now();

        durrr =duration_cast<milliseconds>(expand_end - expand_start).count();
        cout << "Server: first-dimension mul cost = " << durrr << " ms" << endl;



        expand_start  = high_resolution_clock::now();

        for (uint32_t jj = 0; jj < first_dim_intermediate_cts.size(); jj++) {


            //evaluator_->transform_from_ntt_inplace(intermediateCtxts[jj]);
            poc_nfllib_intt_ct(first_dim_intermediate_cts[jj],newcontext_);

        }

        expand_end  = high_resolution_clock::now();
        durrr =  duration_cast<milliseconds>(expand_end - expand_start).count();
        cout << "Server: INTT after first dimension = " << durrr << " ms" << endl;


    }


    uint64_t  new_dimension_size=0, logsize;
    if(nvec.size()>1) {

        for (uint32_t i = 1; i < nvec.size(); i++) {
            new_dimension_size = new_dimension_size + nvec[i];
        }

//        logsize = ceil(log2(new_dimension_size));


    }

    //testing starts from here
    vector<GSWCiphertext> CtMuxBits;
    int total_dim_size = new_dimension_size;

    logsize = ceil(log2(total_dim_size*pir_params_.gsw_decomp_size));


    //int decomp_size = newcontext_->first_context_data()->total_coeff_modulus_bit_count() / pir_params_.gsw_base;
    int decomp_size = pir_params_.gsw_decomp_size ;
    int sk_decomp_size = newcontext_->first_context_data()->total_coeff_modulus_bit_count() / pir_params_.secret_base;
    CtMuxBits.resize(total_dim_size, GSWCiphertext(2 * decomp_size));
    vector<GSWCiphertext>::iterator gswCiphers_ptr = CtMuxBits.begin();


    //thread_server_expand(gswCiphers_ptr, query[1], newcontext_, 0, decomp_size, size, galoisKeys_,  decomp_size, pir_params_.gsw_base, sk_decomp_size, pir_params_.secret_base, sk_enc_);

    auto expand_start  = high_resolution_clock::now();

    gsw_server_expand_combined(gswCiphers_ptr, query[1], newcontext_, 0, decomp_size, total_dim_size, galoisKeys_,  decomp_size, pir_params_.gsw_base, sk_decomp_size, pir_params_.secret_base, sk_enc_, 1<<logsize);

    for (uint32_t jj = 0; jj < CtMuxBits.size(); jj++)
    {
        poc_nfllib_ntt_gsw(CtMuxBits[jj],newcontext_);
    }


    auto expand_end  = high_resolution_clock::now();
    uint64_t durrr =  duration_cast<milliseconds>(expand_end - expand_start).count();
    cout << "Server: expand after first diemension = " << durrr << " ms" << endl;

    auto remaining_start = std::chrono::high_resolution_clock::now();
    //for remaining dimensions we treat them differently
    uint64_t  previous_dim=0;
    for (uint32_t i = 1; i < nvec.size(); i++){

        uint64_t n_i = nvec[i];

        product /= n_i;
        vector<Ciphertext> intermediateCtxts(product);//output size of this dimension




        for (uint64_t k = 0; k < product; k++) {


            intermediateCtxts[k].resize(newcontext_, newcontext_->first_context_data()->parms_id(), 2);
            std::vector<uint64_t *> rlwe_decom;
            rwle_decompositions(first_dim_intermediate_cts[k], newcontext_, decomp_size, pir_params_.gsw_base, rlwe_decom);
            poc_nfllib_ntt_rlwe_decomp(rlwe_decom);
            poc_nfllib_external_product(CtMuxBits[0 + previous_dim], rlwe_decom, newcontext_, decomp_size, intermediateCtxts[k],1);
            for (auto p : rlwe_decom) {
                free(p);
            }

            for (uint64_t j = 1; j < n_i; j++) {


                Ciphertext temp;
                rlwe_decom.clear();
                rwle_decompositions(first_dim_intermediate_cts[k + j * product], newcontext_, decomp_size, pir_params_.gsw_base, rlwe_decom);
                poc_nfllib_ntt_rlwe_decomp(rlwe_decom);
                temp.resize(newcontext_, newcontext_->first_context_data()->parms_id(), 2);
                poc_nfllib_external_product(CtMuxBits[j + previous_dim], rlwe_decom, newcontext_, decomp_size, temp,1);

                for (auto p : rlwe_decom) {
                    free(p);
                }
                evaluator_->add_inplace(intermediateCtxts[k], temp); // Adds to first component.



            }

        }



        for (uint32_t jj = 0; jj < intermediateCtxts.size(); jj++) {

            poc_nfllib_intt_ct(intermediateCtxts[jj],newcontext_);

        }

        first_dim_intermediate_cts.clear();
        first_dim_intermediate_cts=intermediateCtxts;
        previous_dim=previous_dim+n_i;

    }


    auto remaining_end  = high_resolution_clock::now();
     durrr =  duration_cast<milliseconds>(remaining_end - remaining_start).count();
     cout << "Server: remaining-dimensions dot-products = " << durrr << " ms" << endl;



    auto Total_end  = high_resolution_clock::now();
    durrr =  duration_cast<milliseconds>(Total_end - Total_start).count();
//    cout << "Total" << durrr  << endl;

    // disk_tracker.stop();
    // disk_tracker.print_stats();

    return first_dim_intermediate_cts;
}

void pir_server::set_enc_sk(GSWCiphertext sk_enc) {
    sk_enc_=sk_enc;
}

pir_server::~pir_server() {
    munmap(split_db_data, split_db_data_size);
}